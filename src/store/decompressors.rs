use std::io;

pub(crate) use context::DecompressionContext;
use serde::{Deserialize, Serialize};

use super::Compressor;

/// Decompressor is deserialized from the doc store footer, when opening an index.
#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Decompressor {
    /// No compression
    None,
    /// Use the lz4 decompressor (block format)
    #[cfg(feature = "lz4-compression")]
    Lz4,
    /// Use the zstd decompressor
    #[cfg(feature = "zstd-compression")]
    Zstd,
}

impl From<Compressor> for Decompressor {
    fn from(compressor: Compressor) -> Self {
        match compressor {
            Compressor::None => Decompressor::None,
            #[cfg(feature = "lz4-compression")]
            Compressor::Lz4 => Decompressor::Lz4,
            #[cfg(feature = "zstd-compression")]
            Compressor::Zstd(_) => Decompressor::Zstd,
        }
    }
}

impl Decompressor {
    pub(crate) fn from_id(id: u8) -> Decompressor {
        match id {
            0 => Decompressor::None,
            #[cfg(feature = "lz4-compression")]
            1 => Decompressor::Lz4,
            #[cfg(feature = "zstd-compression")]
            4 => Decompressor::Zstd,
            _ => panic!("unknown compressor id {id:?}"),
        }
    }

    pub(crate) fn get_id(&self) -> u8 {
        match self {
            Self::None => 0,
            #[cfg(feature = "lz4-compression")]
            Self::Lz4 => 1,
            #[cfg(feature = "zstd-compression")]
            Self::Zstd => 4,
        }
    }

    pub(crate) fn decompress(
        &self,
        compressed_block: &[u8],
        context: &DecompressionContext,
    ) -> io::Result<Vec<u8>> {
        let mut decompressed_block = vec![];
        self.decompress_into(compressed_block, &mut decompressed_block, context)?;
        Ok(decompressed_block)
    }

    #[inline]
    pub(crate) fn decompress_into(
        &self,
        compressed: &[u8],
        decompressed: &mut Vec<u8>,
        _context: &DecompressionContext,
    ) -> io::Result<()> {
        match self {
            Self::None => {
                decompressed.clear();
                decompressed.extend_from_slice(compressed);
                Ok(())
            }
            #[cfg(feature = "lz4-compression")]
            Self::Lz4 => super::compression_lz4_block::decompress(compressed, decompressed),
            #[cfg(feature = "zstd-compression")]
            Self::Zstd => _context.decompress_zstd(compressed, decompressed),
        }
    }
}

#[cfg(feature = "zstd-compression")]
mod context {
    use std::io;
    use std::sync::{Arc, Mutex};

    use zstd::dict::DecoderDictionary;
    use zstd::zstd_safe::DCtx;

    use crate::store::{compression_zstd_block, dictionary_cache};

    pub(crate) struct DecompressionContext {
        zstd: Mutex<Vec<DCtx<'static>>>,
        prepared_dictionary: Option<DecoderDictionary<'static>>,
    }

    impl DecompressionContext {
        pub(crate) fn load(
            directory: &dyn crate::Directory,
            path: Option<&str>,
        ) -> io::Result<Arc<Self>> {
            if let Some(context) = path.and_then(dictionary_cache::get_context) {
                return Ok(context);
            }
            let dictionary = crate::store::resolve_segment_dictionary(directory, path)?;
            let context = Arc::new(Self::new(dictionary));
            if let Some(path) = path {
                return Ok(dictionary_cache::insert_context(path, context));
            }
            Ok(context)
        }

        pub(crate) fn new(dictionary: Option<Arc<[u8]>>) -> Self {
            Self {
                prepared_dictionary: dictionary.as_deref().map(DecoderDictionary::copy),
                zstd: Mutex::new(Vec::new()),
            }
        }

        pub(super) fn decompress_zstd(
            &self,
            compressed: &[u8],
            decompressed: &mut Vec<u8>,
        ) -> io::Result<()> {
            let context = self.zstd.lock().unwrap().pop();
            let mut context = context.unwrap_or_else(DCtx::create);
            let result = compression_zstd_block::decompress(
                compressed,
                decompressed,
                &mut context,
                self.prepared_dictionary.as_ref(),
            );
            self.zstd.lock().unwrap().push(context);
            result
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::store::Decompressor;

        #[test]
        fn prepared_cache_trusts_content_addressed_names() -> io::Result<()> {
            use crate::directory::RamDirectory;
            use crate::store::ZstdDictionary;
            use crate::Directory;

            let directory = RamDirectory::create();
            let path = ZstdDictionary::seed(
                &directory,
                Arc::from(b"prepared dictionary filename trust fixture".as_slice()),
            )?
            .path()
            .to_string();
            let dictionary = b"dictionary bytes that do not match the filename hash";
            directory.atomic_write(
                std::path::Path::new(&path),
                &compression_zstd_block::compress_whole(dictionary)?,
            )?;
            let context = DecompressionContext::load(&directory, Some(&path))?;
            let data = dictionary.repeat(100);
            let mut compressed = Vec::new();
            compression_zstd_block::compress(
                &data,
                &mut compressed,
                &mut compression_zstd_block::new_compressor(
                    Some(3),
                    Some(dictionary),
                    &Default::default(),
                )?,
            )?;
            assert_eq!(Decompressor::Zstd.decompress(&compressed, &context)?, data);
            Ok(())
        }

        #[test]
        fn later_readers_reuse_prepared_dictionary_after_raw_bytes_are_released() -> io::Result<()>
        {
            use crate::directory::RamDirectory;
            use crate::store::ZstdDictionary;
            use crate::Directory;

            let directory = RamDirectory::create();
            let dictionary = b"dictionary unique to prepared-context cache lifetime test";
            let path = ZstdDictionary::seed(&directory, Arc::from(dictionary.as_slice()))?
                .path()
                .to_string();
            let first = DecompressionContext::load(&directory, Some(&path))?;
            assert!(dictionary_cache::get(&path).is_none());
            let data = dictionary.repeat(1000);
            let mut compressed = Vec::new();
            compression_zstd_block::compress(
                &data,
                &mut compressed,
                &mut compression_zstd_block::new_compressor(
                    Some(3),
                    Some(dictionary),
                    &Default::default(),
                )?,
            )?;
            assert_eq!(Decompressor::Zstd.decompress(&compressed, &first)?, data);
            directory.delete(std::path::Path::new(&path)).unwrap();
            let later = DecompressionContext::load(&directory, Some(&path))?;
            assert!(Arc::ptr_eq(&first, &later));
            drop(first);
            assert_eq!(Decompressor::Zstd.decompress(&compressed, &later)?, data);
            let weak = Arc::downgrade(&later);
            drop(later);
            assert!(weak.upgrade().is_none());
            let result = DecompressionContext::load(&directory, Some(&path));
            assert!(matches!(result, Err(error) if error.kind() == io::ErrorKind::NotFound));
            Ok(())
        }

        #[test]
        fn reused_zstd_context_handles_independent_blocks_and_errors() -> io::Result<()> {
            let dictionary = b"shared dictionary with repeated words and values";
            for dictionary in [None, Some(dictionary.as_slice())] {
                let dictionary_bytes = dictionary.map(Arc::<[u8]>::from);
                let weak_dictionary = dictionary_bytes.as_ref().map(Arc::downgrade);
                let context = DecompressionContext::new(dictionary_bytes);
                if let Some(dictionary) = &weak_dictionary {
                    assert!(dictionary.upgrade().is_none());
                }
                let mut compressed = Vec::new();
                for data in [
                    b"shared dictionary with repeated words and values".repeat(1000),
                    b"different block contents".repeat(17),
                    Vec::new(),
                    b"shared dictionary with repeated words and values".repeat(2000),
                ] {
                    compression_zstd_block::compress(
                        &data,
                        &mut compressed,
                        &mut compression_zstd_block::new_compressor(
                            Some(3),
                            dictionary,
                            &Default::default(),
                        )?,
                    )?;
                    assert_eq!(Decompressor::Zstd.decompress(&compressed, &context)?, data);
                    assert!(Decompressor::Zstd
                        .decompress(&compressed[..3], &context)
                        .is_err());
                    let mut corrupt = compressed.clone();
                    corrupt[4] ^= 0xff;
                    assert!(Decompressor::Zstd.decompress(&corrupt, &context).is_err());
                    if dictionary.is_some() {
                        corrupt.copy_from_slice(&compressed);
                        *corrupt.last_mut().unwrap() ^= 1;
                        assert!(Decompressor::Zstd.decompress(&corrupt, &context).is_err());
                    }
                    assert_eq!(Decompressor::Zstd.decompress(&compressed, &context)?, data);
                    assert_eq!(context.zstd.lock().unwrap().len(), 1);
                }
            }
            Ok(())
        }

        #[test]
        fn shared_zstd_context_supports_parallel_blocks() -> io::Result<()> {
            let dictionary = b"shared dictionary with repeated words and values";
            let context = DecompressionContext::new(Some(Arc::from(dictionary.as_slice())));
            let blocks = (0..8)
                .map(|i| {
                    let data = format!("shared dictionary with repeated words and values {i}")
                        .repeat(1000);
                    let mut compressed = Vec::new();
                    compression_zstd_block::compress(
                        data.as_bytes(),
                        &mut compressed,
                        &mut compression_zstd_block::new_compressor(
                            Some(3),
                            Some(dictionary),
                            &Default::default(),
                        )?,
                    )?;
                    Ok((data, compressed))
                })
                .collect::<io::Result<Vec<_>>>()?;
            let barrier = std::sync::Barrier::new(blocks.len());
            std::thread::scope(|scope| {
                let handles = blocks
                    .iter()
                    .map(|(data, compressed)| {
                        let context = &context;
                        let barrier = &barrier;
                        scope.spawn(move || -> io::Result<()> {
                            barrier.wait();
                            for _ in 0..100 {
                                assert_eq!(
                                    Decompressor::Zstd.decompress(compressed, context)?,
                                    data.as_bytes()
                                );
                            }
                            Ok(())
                        })
                    })
                    .collect::<Vec<_>>();
                for handle in handles {
                    handle.join().unwrap()?;
                }
                Ok(())
            })
        }
    }
}

#[cfg(not(feature = "zstd-compression"))]
mod context {
    use std::io;
    use std::sync::Arc;

    pub(crate) struct DecompressionContext;

    impl DecompressionContext {
        pub(crate) fn load(
            directory: &dyn crate::Directory,
            path: Option<&str>,
        ) -> io::Result<Arc<Self>> {
            crate::store::resolve_segment_dictionary(directory, path)?;
            Ok(Arc::new(Self))
        }

        pub(crate) fn new(_dictionary: Option<Arc<[u8]>>) -> Self {
            Self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compressor_decompressor_id_test() {
        assert_eq!(Decompressor::from(Compressor::None), Decompressor::None);
        #[cfg(feature = "lz4-compression")]
        assert_eq!(Decompressor::from(Compressor::Lz4), Decompressor::Lz4);
        #[cfg(feature = "zstd-compression")]
        assert_eq!(
            Decompressor::from(Compressor::Zstd(Default::default())),
            Decompressor::Zstd
        );
    }
}
