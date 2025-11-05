use std::collections::HashMap;
use std::ops::BitOrAssign;
use std::sync::{Arc, RwLock};
use std::{fmt, io};

use fnv::FnvHashMap;
use itertools::Itertools;

use crate::directory::error::OpenReadError;
use crate::directory::{CompositeFile, FileSlice};
use crate::error::DataCorruption;
use crate::fastfield::{intersect_alive_bitsets, AliveBitSet, FacetReader, FastFieldReaders};
use crate::fieldnorm::{FieldNormReader, FieldNormReaders};
use crate::index::{InvertedIndexReader, Segment, SegmentComponent, SegmentId};
use crate::json_utils::json_path_sep_to_dot;
use crate::schema::{Field, IndexRecordOption, Schema, Type};
use crate::space_usage::SegmentSpaceUsage;
use crate::store::StoreReader;
use crate::termdict::TermDictionary;
use crate::{DocId, Executor, Opstamp, TantivyError};

/// Entry point to access all of the datastructures of the `Segment`
///
/// - term dictionary
/// - postings
/// - store
/// - fast field readers
/// - field norm reader
///
/// The segment reader has a very low memory footprint,
/// as close to all of the memory data is mmapped.
#[derive(Clone)]
pub struct SegmentReader {
    inv_idx_reader_cache: Arc<RwLock<HashMap<Field, Arc<InvertedIndexReader>>>>,

    segment_id: SegmentId,
    delete_opstamp: Option<Opstamp>,

    max_doc: DocId,
    num_docs: DocId,

    termdict_composite: CompositeFile,
    postings_composite: CompositeFile,
    positions_composite: CompositeFile,
    fast_fields_readers: FastFieldReaders,
    fieldnorm_readers: FieldNormReaders,

    store_file: FileSlice,
    alive_bitset_opt: Option<AliveBitSet>,
    schema: Schema,
}

impl SegmentReader {
    /// Returns the highest document id ever attributed in
    /// this segment + 1.
    pub fn max_doc(&self) -> DocId {
        self.max_doc
    }

    /// Returns the number of alive documents.
    /// Deleted documents are not counted.
    pub fn num_docs(&self) -> DocId {
        self.num_docs
    }

    /// Returns the schema of the index this segment belongs to.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Return the number of documents that have been
    /// deleted in the segment.
    pub fn num_deleted_docs(&self) -> DocId {
        self.max_doc - self.num_docs
    }

    /// Returns true if some of the documents of the segment have been deleted.
    pub fn has_deletes(&self) -> bool {
        self.num_deleted_docs() > 0
    }

    /// Accessor to a segment's fast field reader given a field.
    ///
    /// Returns the u64 fast value reader if the field
    /// is a u64 field indexed as "fast".
    ///
    /// Return a FastFieldNotAvailableError if the field is not
    /// declared as a fast field in the schema.
    ///
    /// # Panics
    /// May panic if the index is corrupted.
    pub fn fast_fields(&self) -> &FastFieldReaders {
        &self.fast_fields_readers
    }

    /// Accessor to the `FacetReader` associated with a given `Field`.
    pub fn facet_reader(&self, field_name: &str) -> crate::Result<FacetReader> {
        let schema = self.schema();
        let field = schema.get_field(field_name)?;
        let field_entry = schema.get_field_entry(field);
        if field_entry.field_type().value_type() != Type::Facet {
            return Err(crate::TantivyError::SchemaError(format!(
                "`{field_name}` is not a facet field.`"
            )));
        }
        let Some(facet_column) = self.fast_fields().str(field_name)? else {
            panic!("Facet Field `{field_name}` is missing. This should not happen");
        };
        Ok(FacetReader::new(facet_column))
    }

    /// Accessor to the segment's `Field norms`'s reader.
    ///
    /// Field norms are the length (in tokens) of the fields.
    /// It is used in the computation of the [TfIdf](https://fulmicoton.gitbooks.io/tantivy-doc/content/tfidf.html).
    ///
    /// They are simply stored as a fast field, serialized in
    /// the `.fieldnorm` file of the segment.
    pub fn get_fieldnorms_reader(&self, field: Field) -> crate::Result<FieldNormReader> {
        self.fieldnorm_readers.get_field(field)?.ok_or_else(|| {
            let field_name = self.schema.get_field_name(field);
            let err_msg = format!(
                "Field norm not found for field {field_name:?}. Was the field set to record norm \
                 during indexing?"
            );
            crate::TantivyError::SchemaError(err_msg)
        })
    }

    #[doc(hidden)]
    pub fn fieldnorms_readers(&self) -> &FieldNormReaders {
        &self.fieldnorm_readers
    }

    /// Accessor to the segment's [`StoreReader`](crate::store::StoreReader).
    ///
    /// `cache_num_blocks` sets the number of decompressed blocks to be cached in an LRU.
    /// The size of blocks is configurable, this should be reflexted in the
    pub fn get_store_reader(&self, cache_num_blocks: usize) -> io::Result<StoreReader> {
        StoreReader::open(self.store_file.clone(), cache_num_blocks)
    }

    /// Async counterpart to [`SegmentReader::get_store_reader`].
    pub async fn get_store_reader_async(
        &self,
        cache_num_blocks: usize,
    ) -> io::Result<StoreReader> {
        StoreReader::open_async(self.store_file.clone(), cache_num_blocks).await
    }

    /// Open a new segment for reading.
    pub fn open(segment: &Segment) -> crate::Result<SegmentReader> {
        Self::open_with_custom_alive_set(segment, None)
    }

    /// Open a new segment for reading.
    pub fn open_with_custom_alive_set(
        segment: &Segment,
        custom_bitset: Option<AliveBitSet>,
    ) -> crate::Result<SegmentReader> {
        let termdict_file = segment.open_read(SegmentComponent::Terms)?;
        let termdict_composite = CompositeFile::open(&termdict_file)?;

        let store_file = segment.open_read(SegmentComponent::Store)?;

        crate::fail_point!("SegmentReader::open#middle");

        let postings_file = segment.open_read(SegmentComponent::Postings)?;
        let postings_composite = CompositeFile::open(&postings_file)?;

        let positions_composite = {
            if let Ok(positions_file) = segment.open_read(SegmentComponent::Positions) {
                CompositeFile::open(&positions_file)?
            } else {
                CompositeFile::empty()
            }
        };

        let schema = segment.schema();

        let fast_fields_data = segment.open_read(SegmentComponent::FastFields)?;
        let fast_fields_readers = FastFieldReaders::open(fast_fields_data, schema.clone())?;
        let fieldnorm_data = segment.open_read(SegmentComponent::FieldNorms)?;
        let fieldnorm_readers = FieldNormReaders::open(fieldnorm_data)?;

        let original_bitset = if segment.meta().has_deletes() {
            let alive_doc_file_slice = segment.open_read(SegmentComponent::Delete)?;
            let alive_doc_data = alive_doc_file_slice.read_bytes()?;
            Some(AliveBitSet::open(alive_doc_data))
        } else {
            None
        };

        let alive_bitset_opt = intersect_alive_bitset(original_bitset, custom_bitset);

        let max_doc = segment.meta().max_doc();
        let num_docs = alive_bitset_opt
            .as_ref()
            .map(|alive_bitset| alive_bitset.num_alive_docs() as u32)
            .unwrap_or(max_doc);

        Ok(SegmentReader {
            inv_idx_reader_cache: Default::default(),
            num_docs,
            max_doc,
            termdict_composite,
            postings_composite,
            fast_fields_readers,
            fieldnorm_readers,
            segment_id: segment.id(),
            delete_opstamp: segment.meta().delete_opstamp(),
            store_file,
            alive_bitset_opt,
            positions_composite,
            schema,
        })
    }

    /// Async version of `open_with_custom_alive_set`.
    pub async fn open_with_custom_alive_set_async(
        segment: &Segment,
        custom_bitset: Option<AliveBitSet>,
    ) -> crate::Result<SegmentReader> {
        let (
            termdict_file,
            store_file,
            postings_file,
            fast_fields_data,
            fieldnorm_data,
            positions_file_opt,
        ) = futures::try_join!(
            segment.open_read_async(SegmentComponent::Terms),
            segment.open_read_async(SegmentComponent::Store),
            segment.open_read_async(SegmentComponent::Postings),
            segment.open_read_async(SegmentComponent::FastFields),
            segment.open_read_async(SegmentComponent::FieldNorms),
            async {
                match segment.open_read_async(SegmentComponent::Positions).await {
                    Ok(file) => Ok(Some(file)),
                    Err(OpenReadError::FileDoesNotExist(_)) => Ok(None),
                    Err(err) => Err(err),
                }
            }
        )?;

        let schema = segment.schema();

        crate::fail_point!("SegmentReader::open#middle");

        let (
            termdict_composite,
            postings_composite,
            positions_composite,
            fast_fields_readers,
            fieldnorm_readers,
            original_bitset,
        ) = futures::try_join!(
            async {
                Ok::<_, TantivyError>(CompositeFile::open_async(&termdict_file).await?)
            },
            async {
                Ok::<_, TantivyError>(CompositeFile::open_async(&postings_file).await?)
            },
            async {
                if let Some(positions_file) = positions_file_opt {
                    Ok::<_, TantivyError>(CompositeFile::open_async(&positions_file).await?)
                } else {
                    Ok(CompositeFile::empty())
                }
            },
            async {
                Ok::<_, TantivyError>(
                    FastFieldReaders::open_async(fast_fields_data, schema.clone()).await?
                )
            },
            async { FieldNormReaders::open_async(fieldnorm_data).await },
            async {
                if segment.meta().has_deletes() {
                    let alive_doc_file_slice =
                        segment.open_read_async(SegmentComponent::Delete).await?;
                    let alive_doc_data = alive_doc_file_slice.read_bytes_async().await?;
                    Ok::<_, TantivyError>(Some(AliveBitSet::open(alive_doc_data)))
                } else {
                    Ok(None)
                }
            }
        )?;

        let alive_bitset_opt = intersect_alive_bitset(original_bitset, custom_bitset);

        let max_doc = segment.meta().max_doc();
        let num_docs = alive_bitset_opt
            .as_ref()
            .map(|alive_bitset| alive_bitset.num_alive_docs() as u32)
            .unwrap_or(max_doc);

        Ok(SegmentReader {
            inv_idx_reader_cache: Default::default(),
            num_docs,
            max_doc,
            termdict_composite,
            postings_composite,
            fast_fields_readers,
            fieldnorm_readers,
            segment_id: segment.id(),
            delete_opstamp: segment.meta().delete_opstamp(),
            store_file,
            alive_bitset_opt,
            positions_composite,
            schema,
        })
    }

    /// Open a new segment for reading.
    ///
    /// Previously used nested parallelism (rayon::join) but this caused stack overflow
    /// when opening many segments. Now just calls the sequential version.
    /// Parallelism across segments (via executor.map) is sufficient.
    pub fn open_with_custom_alive_set_parallel(
        _executor: &Executor,
        segment: &Segment,
        custom_bitset: Option<AliveBitSet>,
    ) -> crate::Result<SegmentReader> {
        // Don't parallelize component loading within a segment to avoid stack overflow
        // The outer executor.map() already parallelizes across segments
        Self::open_with_custom_alive_set(segment, custom_bitset)
    }

    /// Returns a field reader associated with the field given in argument.
    /// If the field was not present in the index during indexing time,
    /// the InvertedIndexReader is empty.
    ///
    /// The field reader is in charge of iterating through the
    /// term dictionary associated with a specific field,
    /// and opening the posting list associated with any term.
    ///
    /// If the field is not marked as index, a warning is logged and an empty `InvertedIndexReader`
    /// is returned.
    /// Similarly, if the field is marked as indexed but no term has been indexed for the given
    /// index, an empty `InvertedIndexReader` is returned (but no warning is logged).
    pub fn inverted_index(&self, field: Field) -> crate::Result<Arc<InvertedIndexReader>> {
        if let Some(inv_idx_reader) = self
            .inv_idx_reader_cache
            .read()
            .expect("Lock poisoned. This should never happen")
            .get(&field)
        {
            return Ok(Arc::clone(inv_idx_reader));
        }
        let field_entry = self.schema.get_field_entry(field);
        let field_type = field_entry.field_type();
        let record_option_opt = field_type.get_index_record_option();

        if record_option_opt.is_none() {
            warn!("Field {:?} does not seem indexed.", field_entry.name());
        }

        let postings_file_opt = self.postings_composite.open_read(field);

        if postings_file_opt.is_none() || record_option_opt.is_none() {
            // no documents in the segment contained this field.
            // As a result, no data is associated with the inverted index.
            //
            // Returns an empty inverted index.
            let record_option = record_option_opt.unwrap_or(IndexRecordOption::Basic);
            return Ok(Arc::new(InvertedIndexReader::empty(record_option)));
        }

        let record_option = record_option_opt.unwrap();
        let postings_file = postings_file_opt.unwrap();

        let termdict_file: FileSlice =
            self.termdict_composite.open_read(field).ok_or_else(|| {
                DataCorruption::comment_only(format!(
                    "Failed to open field {:?}'s term dictionary in the composite file. Has the \
                     schema been modified?",
                    field_entry.name()
                ))
            })?;

        let positions_file = self.positions_composite.open_read(field).ok_or_else(|| {
            let error_msg = format!(
                "Failed to open field {:?}'s positions in the composite file. Has the schema been \
                 modified?",
                field_entry.name()
            );
            DataCorruption::comment_only(error_msg)
        })?;

        let inv_idx_reader = Arc::new(InvertedIndexReader::new(
            TermDictionary::open(termdict_file)?,
            postings_file,
            positions_file,
            record_option,
        )?);

        // by releasing the lock in between, we may end up opening the inverting index
        // twice, but this is fine.
        self.inv_idx_reader_cache
            .write()
            .expect("Field reader cache lock poisoned. This should never happen.")
            .insert(field, Arc::clone(&inv_idx_reader));

        Ok(inv_idx_reader)
    }

    /// Returns the list of fields that have been indexed in the segment.
    /// The field list includes the field defined in the schema as well as the fields
    /// that have been indexed as a part of a JSON field.
    /// The returned field name is the full field name, including the name of the JSON field.
    ///
    /// The returned field names can be used in queries.
    ///
    /// Notice: If your data contains JSON fields this is **very expensive**, as it requires
    /// browsing through the inverted index term dictionary and the columnar field dictionary.
    ///
    /// Disclaimer: Some fields may not be listed here. For instance, if the schema contains a json
    /// field that is not indexed nor a fast field but is stored, it is possible for the field
    /// to not be listed.
    pub fn fields_metadata(&self) -> crate::Result<Vec<FieldMetadata>> {
        let mut indexed_fields: Vec<FieldMetadata> = Vec::new();
        let mut map_to_canonical = FnvHashMap::default();
        for (field, field_entry) in self.schema().fields() {
            let field_name = field_entry.name().to_string();
            let is_indexed = field_entry.is_indexed();

            if is_indexed {
                let is_json = field_entry.field_type().value_type() == Type::Json;
                if is_json {
                    let inv_index = self.inverted_index(field)?;
                    let encoded_fields_in_index = inv_index.list_encoded_fields()?;
                    let mut build_path = |field_name: &str, mut json_path: String| {
                        // In this case we need to map the potential fast field to the field name
                        // accepted by the query parser.
                        let create_canonical =
                            !field_entry.is_expand_dots_enabled() && json_path.contains('.');
                        if create_canonical {
                            // Without expand dots enabled dots need to be escaped.
                            let escaped_json_path = json_path.replace('.', "\\.");
                            let full_path = format!("{}.{}", field_name, escaped_json_path);
                            let full_path_unescaped = format!("{}.{}", field_name, &json_path);
                            map_to_canonical.insert(full_path_unescaped, full_path.to_string());
                            full_path
                        } else {
                            // With expand dots enabled, we can use '.' instead of '\u{1}'.
                            json_path_sep_to_dot(&mut json_path);
                            format!("{}.{}", field_name, json_path)
                        }
                    };
                    indexed_fields.extend(
                        encoded_fields_in_index
                            .into_iter()
                            .map(|(name, typ)| (build_path(&field_name, name), typ))
                            .map(|(field_name, typ)| FieldMetadata {
                                indexed: true,
                                stored: false,
                                field_name,
                                fast: false,
                                typ,
                            }),
                    );
                } else {
                    indexed_fields.push(FieldMetadata {
                        indexed: true,
                        stored: false,
                        field_name: field_name.to_string(),
                        fast: false,
                        typ: field_entry.field_type().value_type(),
                    });
                }
            }
        }
        let mut fast_fields: Vec<FieldMetadata> = self
            .fast_fields()
            .columnar()
            .iter_columns()?
            .map(|(mut field_name, handle)| {
                json_path_sep_to_dot(&mut field_name);
                // map to canonical path, to avoid similar but different entries.
                // Eventually we should just accept '.' seperated for all cases.
                let field_name = map_to_canonical
                    .get(&field_name)
                    .unwrap_or(&field_name)
                    .to_string();
                FieldMetadata {
                    indexed: false,
                    stored: false,
                    field_name,
                    fast: true,
                    typ: Type::from(handle.column_type()),
                }
            })
            .collect();
        // Since the type is encoded differently in the fast field and in the inverted index,
        // the order of the fields is not guaranteed to be the same. Therefore, we sort the fields.
        // If we are sure that the order is the same, we can remove this sort.
        indexed_fields.sort_unstable();
        fast_fields.sort_unstable();
        let merged = merge_field_meta_data(vec![indexed_fields, fast_fields], &self.schema);

        Ok(merged)
    }

    /// Returns the segment id
    pub fn segment_id(&self) -> SegmentId {
        self.segment_id
    }

    /// Returns the delete opstamp
    pub fn delete_opstamp(&self) -> Option<Opstamp> {
        self.delete_opstamp
    }

    /// Returns the bitset representing the alive `DocId`s.
    pub fn alive_bitset(&self) -> Option<&AliveBitSet> {
        self.alive_bitset_opt.as_ref()
    }

    /// Returns true if the `doc` is marked
    /// as deleted.
    pub fn is_deleted(&self, doc: DocId) -> bool {
        self.alive_bitset()
            .map(|alive_bitset| alive_bitset.is_deleted(doc))
            .unwrap_or(false)
    }

    /// Returns an iterator that will iterate over the alive document ids
    pub fn doc_ids_alive(&self) -> Box<dyn Iterator<Item = DocId> + '_> {
        if let Some(alive_bitset) = &self.alive_bitset_opt {
            Box::new(alive_bitset.iter_alive())
        } else {
            Box::new(0u32..self.max_doc)
        }
    }

    /// Summarize total space usage of this segment.
    pub fn space_usage(&self) -> io::Result<SegmentSpaceUsage> {
        Ok(SegmentSpaceUsage::new(
            self.num_docs(),
            self.termdict_composite.space_usage(),
            self.postings_composite.space_usage(),
            self.positions_composite.space_usage(),
            self.fast_fields_readers.space_usage(self.schema())?,
            self.fieldnorm_readers.space_usage(),
            self.get_store_reader(0)?.space_usage(),
            self.alive_bitset_opt
                .as_ref()
                .map(AliveBitSet::space_usage)
                .unwrap_or_default(),
        ))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
/// FieldMetadata
pub struct FieldMetadata {
    /// The field name
    // Notice: Don't reorder the declaration of 1.field_name 2.typ, as it is used for ordering by
    // field_name then typ.
    pub field_name: String,
    /// The field type
    // Notice: Don't reorder the declaration of 1.field_name 2.typ, as it is used for ordering by
    // field_name then typ.
    pub typ: Type,
    /// Is the field indexed for search
    pub indexed: bool,
    /// Is the field stored in the doc store
    pub stored: bool,
    /// Is the field stored in the columnar storage
    pub fast: bool,
}
impl BitOrAssign for FieldMetadata {
    fn bitor_assign(&mut self, rhs: Self) {
        assert!(self.field_name == rhs.field_name);
        assert!(self.typ == rhs.typ);
        self.indexed |= rhs.indexed;
        self.stored |= rhs.stored;
        self.fast |= rhs.fast;
    }
}

// Maybe too slow for the high cardinality case
fn is_field_stored(field_name: &str, schema: &Schema) -> bool {
    schema
        .find_field(field_name)
        .map(|(field, _path)| schema.get_field_entry(field).is_stored())
        .unwrap_or(false)
}

/// Helper to merge the field metadata from multiple segments.
pub fn merge_field_meta_data(
    field_metadatas: Vec<Vec<FieldMetadata>>,
    schema: &Schema,
) -> Vec<FieldMetadata> {
    let mut merged_field_metadata = Vec::new();
    for (_key, mut group) in &field_metadatas
        .into_iter()
        .kmerge_by(|left, right| left < right)
        // TODO: Remove allocation
        .group_by(|el| (el.field_name.to_string(), el.typ))
    {
        let mut merged: FieldMetadata = group.next().unwrap();
        for el in group {
            merged |= el;
        }
        // Currently is_field_stored is maybe too slow for the high cardinality case
        merged.stored = is_field_stored(&merged.field_name, schema);
        merged_field_metadata.push(merged);
    }
    merged_field_metadata
}

fn intersect_alive_bitset(
    left_opt: Option<AliveBitSet>,
    right_opt: Option<AliveBitSet>,
) -> Option<AliveBitSet> {
    match (left_opt, right_opt) {
        (Some(left), Some(right)) => {
            assert_eq!(left.bitset().max_value(), right.bitset().max_value());
            Some(intersect_alive_bitsets(left, right))
        }
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

impl fmt::Debug for SegmentReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SegmentReader({:?})", self.segment_id)
    }
}

// This is a cheap trick to reference the segment in the `open_with_custom_alive_set_parallel`
// function without having to copy the segment
struct SegmentRef {
    segment: &'static Segment,
}

impl SegmentRef {
    fn new(segment: &Segment) -> Self {
        let leaked_segment = unsafe { std::mem::transmute(segment) };
        Self {
            segment: leaked_segment,
        }
    }
}

impl std::ops::Deref for SegmentRef {
    type Target = Segment;
    fn deref(&self) -> &Self::Target {
        self.segment
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::index::Index;
    use crate::schema::{SchemaBuilder, Term, STORED, TEXT};
    use crate::IndexWriter;

    /// Trait to abstract over sync and async segment opening for testing.
    /// This allows us to test both implementations with the same test logic.
    trait SegmentOpener {
        fn open_segment(&self, segment: &Segment) -> crate::Result<SegmentReader>;
    }

    /// Synchronous implementation - uses the regular sync API
    struct SyncSegmentOpener;

    impl SegmentOpener for SyncSegmentOpener {
        fn open_segment(&self, segment: &Segment) -> crate::Result<SegmentReader> {
            SegmentReader::open(segment)
        }
    }

    #[cfg(feature = "tokio")]
    /// Async implementation wrapped in sync interface - uses runtime.block_on
    struct AsyncSegmentOpener {
        runtime: tokio::runtime::Runtime,
    }

    #[cfg(feature = "tokio")]
    impl AsyncSegmentOpener {
        fn new() -> Self {
            Self {
                runtime: tokio::runtime::Runtime::new().unwrap(),
            }
        }
    }

    #[cfg(feature = "tokio")]
    impl SegmentOpener for AsyncSegmentOpener {
        fn open_segment(&self, segment: &Segment) -> crate::Result<SegmentReader> {
            self.runtime
                .block_on(SegmentReader::open_with_custom_alive_set_async(segment, None))
        }
    }

    #[test]
    fn test_merge_field_meta_data_same() {
        let schema = SchemaBuilder::new().build();
        let field_metadata1 = FieldMetadata {
            field_name: "a".to_string(),
            typ: crate::schema::Type::Str,
            indexed: true,
            stored: false,
            fast: true,
        };
        let field_metadata2 = FieldMetadata {
            field_name: "a".to_string(),
            typ: crate::schema::Type::Str,
            indexed: true,
            stored: false,
            fast: true,
        };
        let res = merge_field_meta_data(
            vec![vec![field_metadata1.clone()], vec![field_metadata2]],
            &schema,
        );
        assert_eq!(res, vec![field_metadata1]);
    }
    #[test]
    fn test_merge_field_meta_data_different() {
        let schema = SchemaBuilder::new().build();
        let field_metadata1 = FieldMetadata {
            field_name: "a".to_string(),
            typ: crate::schema::Type::Str,
            indexed: false,
            stored: false,
            fast: true,
        };
        let field_metadata2 = FieldMetadata {
            field_name: "b".to_string(),
            typ: crate::schema::Type::Str,
            indexed: false,
            stored: false,
            fast: true,
        };
        let field_metadata3 = FieldMetadata {
            field_name: "a".to_string(),
            typ: crate::schema::Type::Str,
            indexed: true,
            stored: false,
            fast: false,
        };
        let res = merge_field_meta_data(
            vec![
                vec![field_metadata1.clone(), field_metadata2.clone()],
                vec![field_metadata3],
            ],
            &schema,
        );
        let field_metadata_expected1 = FieldMetadata {
            field_name: "a".to_string(),
            typ: crate::schema::Type::Str,
            indexed: true,
            stored: false,
            fast: true,
        };
        assert_eq!(res, vec![field_metadata_expected1, field_metadata2.clone()]);
    }
    #[test]
    fn test_merge_field_meta_data_merge() {
        use pretty_assertions::assert_eq;
        let get_meta_data = |name: &str, typ: Type| FieldMetadata {
            field_name: name.to_string(),
            typ,
            indexed: false,
            stored: false,
            fast: true,
        };
        let schema = SchemaBuilder::new().build();
        let mut metas = vec![get_meta_data("d", Type::Str), get_meta_data("e", Type::U64)];
        metas.sort();
        let res = merge_field_meta_data(vec![vec![get_meta_data("e", Type::Str)], metas], &schema);
        assert_eq!(
            res,
            vec![
                get_meta_data("d", Type::Str),
                get_meta_data("e", Type::Str),
                get_meta_data("e", Type::U64),
            ]
        );
    }
    #[test]
    fn test_merge_field_meta_data_bitxor() {
        let field_metadata1 = FieldMetadata {
            field_name: "a".to_string(),
            typ: crate::schema::Type::Str,
            indexed: false,
            stored: false,
            fast: true,
        };
        let field_metadata2 = FieldMetadata {
            field_name: "a".to_string(),
            typ: crate::schema::Type::Str,
            indexed: true,
            stored: false,
            fast: false,
        };
        let field_metadata_expected = FieldMetadata {
            field_name: "a".to_string(),
            typ: crate::schema::Type::Str,
            indexed: true,
            stored: false,
            fast: true,
        };
        let mut res1 = field_metadata1.clone();
        res1 |= field_metadata2.clone();
        let mut res2 = field_metadata2.clone();
        res2 |= field_metadata1;
        assert_eq!(res1, field_metadata_expected);
        assert_eq!(res2, field_metadata_expected);
    }

    fn test_num_alive_impl(opener: &dyn SegmentOpener) -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        schema_builder.add_text_field("name", TEXT | STORED);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema.clone());
        let name = schema.get_field("name").unwrap();

        {
            let mut index_writer: IndexWriter = index.writer_for_tests()?;
            index_writer.add_document(doc!(name => "tantivy"))?;
            index_writer.add_document(doc!(name => "horse"))?;
            index_writer.add_document(doc!(name => "jockey"))?;
            index_writer.add_document(doc!(name => "cap"))?;
            // we should now have one segment with two docs
            index_writer.delete_term(Term::from_field_text(name, "horse"));
            index_writer.delete_term(Term::from_field_text(name, "cap"));

            // ok, now we should have a deleted doc
            index_writer.commit()?;
        }

        let segments = index.searchable_segments()?;
        let segment_reader = opener.open_segment(&segments[0])?;
        assert_eq!(2, segment_reader.num_docs());
        assert_eq!(4, segment_reader.max_doc());
        Ok(())
    }

    #[test]
    fn test_num_alive() -> crate::Result<()> {
        test_num_alive_impl(&SyncSegmentOpener)
    }

    #[test]
    #[cfg(feature = "tokio")]
    fn test_num_alive_async() -> crate::Result<()> {
        test_num_alive_impl(&AsyncSegmentOpener::new())
    }
    fn test_alive_docs_iterator_impl(opener: &dyn SegmentOpener) -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        schema_builder.add_text_field("name", TEXT | STORED);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema.clone());
        let name = schema.get_field("name").unwrap();

        {
            let mut index_writer: IndexWriter = index.writer_for_tests()?;
            index_writer.add_document(doc!(name => "tantivy"))?;
            index_writer.add_document(doc!(name => "horse"))?;
            index_writer.add_document(doc!(name => "jockey"))?;
            index_writer.add_document(doc!(name => "cap"))?;
            // we should now have one segment with two docs
            index_writer.commit()?;
        }

        {
            let mut index_writer2: IndexWriter = index.writer(50_000_000)?;
            index_writer2.delete_term(Term::from_field_text(name, "horse"));
            index_writer2.delete_term(Term::from_field_text(name, "cap"));

            // ok, now we should have a deleted doc
            index_writer2.commit()?;
        }

        let segments = index.searchable_segments()?;
        let segment_reader = opener.open_segment(&segments[0])?;
        let docs: Vec<DocId> = segment_reader.doc_ids_alive().collect();
        assert_eq!(vec![0u32, 2u32], docs);
        Ok(())
    }

    #[test]
    fn test_alive_docs_iterator() -> crate::Result<()> {
        test_alive_docs_iterator_impl(&SyncSegmentOpener)
    }

    #[test]
    #[cfg(feature = "tokio")]
    fn test_alive_docs_iterator_async() -> crate::Result<()> {
        test_alive_docs_iterator_impl(&AsyncSegmentOpener::new())
    }

    #[test]
    fn test_parallel_segment_opening_deadlock() -> crate::Result<()> {
        use std::sync::Arc;

        // Create an index with multiple segments (more than thread pool size)
        let mut schema_builder = Schema::builder();
        schema_builder.add_text_field("text", TEXT | STORED);
        let schema = schema_builder.build();
        let text_field = schema.get_field("text").unwrap();

        // Use a small thread pool to make deadlock more likely
        let executor = Arc::new(crate::Executor::multi_thread(4, "deadlock-test-")?);
        let index = Index::create_in_ram(schema.clone());

        // Create 8 segments (more than pool size of 4)
        {
            let mut index_writer = index.writer_for_tests()?;
            for seg_num in 0..8 {
                // Add a few docs to each segment
                for doc_num in 0..10 {
                    index_writer.add_document(
                        doc!(text_field => format!("segment {} doc {}", seg_num, doc_num)),
                    )?;
                }
                // Commit to create a new segment
                index_writer.commit()?;
            }
        }

        // Get all segments
        let segments = index.searchable_segments()?;
        assert_eq!(segments.len(), 8, "Should have 8 segments");

        // Now try to open all segments in parallel using the same executor
        // This mimics what open_segment_readers() does
        let executor_clone = executor.clone();
        let result = executor.map(
            |segment| {
                // This calls open_with_custom_alive_set_parallel which:
                // 1. Spawns 6 file-loading tasks via executor.spawn()
                // 2. Immediately blocks waiting for them via fut()?
                // Since we're already inside executor.map(), this is nested parallelism
                SegmentReader::open_with_custom_alive_set_parallel(&executor_clone, segment, None)
            },
            segments.iter(),
        );

        // If you run this test with --ignored, it will deadlock here
        // All 4 workers will be blocked in step 2 above, waiting for the 6 spawned tasks
        // But those spawned tasks need workers to execute, and all workers are blocked!
        result?;

        Ok(())
    }

    #[test]
    fn test_sequential_segment_opening_no_deadlock() -> crate::Result<()> {
        use std::sync::Arc;

        // Create an index with multiple segments
        let mut schema_builder = Schema::builder();
        schema_builder.add_text_field("text", TEXT | STORED);
        let schema = schema_builder.build();
        let text_field = schema.get_field("text").unwrap();

        let executor = Arc::new(crate::Executor::multi_thread(4, "safe-test-")?);
        let index = Index::create_in_ram(schema.clone());

        // Create 8 segments
        {
            let mut index_writer = index.writer_for_tests()?;
            for seg_num in 0..8 {
                for doc_num in 0..5 {
                    index_writer.add_document(
                        doc!(text_field => format!("segment {} doc {}", seg_num, doc_num)),
                    )?;
                }
                index_writer.commit()?;
            }
        }

        let segments = index.searchable_segments()?;
        assert_eq!(segments.len(), 8);

        // Use the executor for outer parallelism, but use non-parallel segment opening
        let result = executor.map(
            |segment| {
                // Use the non-parallel version - no nested executor calls!
                SegmentReader::open(segment)
            },
            segments.iter(),
        );

        // This should complete successfully because we're not nesting executor calls
        let segment_readers = result?;
        assert_eq!(segment_readers.len(), 8);
        println!(
            "✓ Successfully opened {} segments without deadlock",
            segment_readers.len()
        );

        Ok(())
    }

    #[tokio::test]
    #[cfg(feature = "tokio")]
    async fn test_async_segment_opening_concurrent() -> crate::Result<()> {
        // Create an index with multiple segments
        let mut schema_builder = Schema::builder();
        schema_builder.add_text_field("text", TEXT | STORED);
        let schema = schema_builder.build();
        let text_field = schema.get_field("text").unwrap();

        let index = Index::create_in_ram(schema.clone());

        // Create 8 segments
        {
            let mut index_writer = index.writer_for_tests()?;
            for seg_num in 0..8 {
                for doc_num in 0..5 {
                    index_writer.add_document(
                        doc!(text_field => format!("segment {} doc {}", seg_num, doc_num)),
                    )?;
                }
                index_writer.commit()?;
            }
        }

        let segments = index.searchable_segments()?;
        assert_eq!(segments.len(), 8);

        // Open all segments concurrently using async
        let segment_readers = futures::future::try_join_all(
            segments
                .iter()
                .map(|segment| SegmentReader::open_with_custom_alive_set_async(segment, None)),
        )
        .await?;

        assert_eq!(segment_readers.len(), 8);

        // Verify all segments are correctly opened
        for (i, reader) in segment_readers.iter().enumerate() {
            assert_eq!(reader.num_docs(), 5, "Segment {} should have 5 docs", i);
        }

        Ok(())
    }

    #[test]
    #[cfg(feature = "tokio")]
    fn test_async_store_reader() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        schema_builder.add_text_field("text", TEXT | STORED);
        let schema = schema_builder.build();
        let text_field = schema.get_field("text").unwrap();

        let index = Index::create_in_ram(schema.clone());

        {
            let mut index_writer = index.writer_for_tests()?;
            index_writer.add_document(doc!(text_field => "test document"))?;
            index_writer.commit()?;
        }

        let segments = index.searchable_segments()?;
        let segment = &segments[0];

        // Test async store reader opening
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let segment_reader = runtime.block_on(
            SegmentReader::open_with_custom_alive_set_async(segment, None)
        )?;

        let store_reader = runtime.block_on(
            segment_reader.get_store_reader_async(10)
        )?;

        // Verify we can read from it
        let doc: crate::TantivyDocument = store_reader.get(0)?;
        assert!(doc.get_first(text_field).is_some());

        Ok(())
    }
}
