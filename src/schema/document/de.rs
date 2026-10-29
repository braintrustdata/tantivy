//! Document binary deserialization API
//!
//! The deserialization API is strongly inspired by serde's API but with
//! some tweaks, mostly around some of the types being concrete (errors)
//! and some more specific types being visited (Ips, datetime, etc...)
//!
//! The motivation behind this API is to provide a easy to implement and
//! efficient way of deserializing a potentially arbitrarily nested object.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Display;
use std::io;
use std::io::Read;
use std::marker::PhantomData;
use std::net::Ipv6Addr;
use std::sync::Arc;

use columnar::MonotonicallyMappableToU128;
use common::{u64_to_f64, BinarySerializable, DateTime, VInt};

use super::{OwnedValue, Value};
use crate::schema::document::type_codes;
use crate::schema::{Facet, Field};
use crate::tokenizer::PreTokenizedString;

#[derive(Debug, thiserror::Error, Clone)]
/// An error which occurs while attempting to deserialize a given value
/// by using the provided value visitor.
pub enum DeserializeError {
    #[error("Unsupported Type: {0:?} cannot be deserialized from the given visitor")]
    /// The value cannot be deserialized from the given type.
    UnsupportedType(ValueType),
    #[error("Type Mismatch: Expected {expected:?} but found {actual:?}")]
    /// The value cannot be deserialized from the given type.
    TypeMismatch {
        /// The expected value type.
        expected: ValueType,
        /// The actual value type read.
        actual: ValueType,
    },
    #[error("The value could not be read: {0}")]
    /// The value was unable to be read due to the error.
    CorruptedValue(Arc<io::Error>),
    #[error("{0}")]
    /// A custom error message.
    Custom(String),
}

impl DeserializeError {
    /// Creates a new custom deserialize error.
    pub fn custom(msg: impl Display) -> Self {
        Self::Custom(msg.to_string())
    }
}

impl From<io::Error> for DeserializeError {
    fn from(error: io::Error) -> Self {
        Self::CorruptedValue(Arc::new(error))
    }
}

/// The core trait for deserializing a document.
///
/// TODO: Improve docs
pub trait DocumentDeserialize: Sized {
    /// Attempts to deserialize Self from a given document deserializer.
    fn deserialize<'de, D>(deserializer: D) -> Result<Self, DeserializeError>
    where D: DocumentDeserializer<'de>;
}

/// A deserializer that can walk through each entry in the document.
pub trait DocumentDeserializer<'de> {
    /// A indicator as to how many values are in the document.
    ///
    /// This can be used to pre-allocate entries but should not
    /// be depended on as a fixed size.
    fn size_hint(&self) -> usize;

    /// Attempts to deserialize the next field in the document.
    fn next_field<V: ValueDeserialize>(&mut self) -> Result<Option<(Field, V)>, DeserializeError>;

    /// Deserializes the next field using caller-provided state.
    fn next_field_seed<S: ValueDeserializeSeed>(
        &mut self,
        _seed: S,
    ) -> Result<Option<(Field, S::Value)>, DeserializeError> {
        Err(DeserializeError::custom(
            "Document deserializer does not support seeds",
        ))
    }
}

/// Deserializes a value using caller-provided state.
pub trait ValueDeserializeSeed {
    /// The value produced by this seed.
    type Value;

    /// Deserializes a value with this seed's state.
    fn deserialize<'de, D>(self, deserializer: D) -> Result<Self::Value, DeserializeError>
    where D: ValueDeserializer<'de>;
}

impl<T: ValueDeserialize> ValueDeserializeSeed for PhantomData<T> {
    type Value = T;

    fn deserialize<'de, D>(self, deserializer: D) -> Result<T, DeserializeError>
    where D: ValueDeserializer<'de> {
        T::deserialize(deserializer)
    }
}

/// The core trait for deserializing values.
pub trait ValueDeserialize: Sized {
    /// Attempts to deserialize Self from a given value deserializer.
    fn deserialize<'de, D>(deserializer: D) -> Result<Self, DeserializeError>
    where D: ValueDeserializer<'de>;
}

/// A value deserializer.
pub trait ValueDeserializer<'de> {
    /// Attempts to deserialize a null value from the deserializer.
    fn deserialize_null(self) -> Result<(), DeserializeError>;

    /// Attempts to deserialize a string value from the deserializer.
    fn deserialize_string(self) -> Result<String, DeserializeError>;

    /// Attempts to deserialize a u64 value from the deserializer.
    fn deserialize_u64(self) -> Result<u64, DeserializeError>;

    /// Attempts to deserialize an i64 value from the deserializer.
    fn deserialize_i64(self) -> Result<i64, DeserializeError>;

    /// Attempts to deserialize a f64 value from the deserializer.
    fn deserialize_f64(self) -> Result<f64, DeserializeError>;

    /// Attempts to deserialize a datetime value from the deserializer.
    fn deserialize_datetime(self) -> Result<DateTime, DeserializeError>;

    /// Attempts to deserialize a facet value from the deserializer.
    fn deserialize_facet(self) -> Result<Facet, DeserializeError>;

    /// Attempts to deserialize a bytes value from the deserializer.
    fn deserialize_bytes(self) -> Result<Vec<u8>, DeserializeError>;

    /// Attempts to deserialize an IP address value from the deserializer.
    fn deserialize_ip_address(self) -> Result<Ipv6Addr, DeserializeError>;

    /// Attempts to deserialize a bool value from the deserializer.
    fn deserialize_bool(self) -> Result<bool, DeserializeError>;

    /// Attempts to deserialize a pre-tokenized string value from the deserializer.
    fn deserialize_pre_tokenized_string(self) -> Result<PreTokenizedString, DeserializeError>;

    /// Attempts to deserialize the value using a given visitor.
    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, DeserializeError>
    where V: ValueVisitor;
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
/// The type of the value attempting to be deserialized.
pub enum ValueType {
    /// A null value.
    Null,
    /// A string value.
    String,
    /// A u64 value.
    U64,
    /// A i64 value.
    I64,
    /// A f64 value.
    F64,
    /// A datetime value.
    DateTime,
    /// A facet value.
    Facet,
    /// A bytes value.
    Bytes,
    /// A IP address value.
    IpAddr,
    /// A boolean value.
    Bool,
    /// A pre-tokenized string value.
    PreTokStr,
    /// An array of value.
    Array,
    /// A dynamic object value.
    Object,
    /// A JSON object value. Deprecated.
    #[deprecated(note = "We keep this for backwards compatibility, use Object instead")]
    JSONObject,
}

/// A value visitor for deserializing a document value.
///
/// This is strongly inspired by serde but has a few extra types.
///
/// TODO: Improve docs
pub trait ValueVisitor {
    /// The value produced by the visitor.
    type Value;

    #[inline]
    /// Called when the deserializer visits a string value.
    fn visit_null(&self) -> Result<Self::Value, DeserializeError> {
        Err(DeserializeError::UnsupportedType(ValueType::Null))
    }

    #[inline]
    /// Visits a string borrowed from the input. Defaults to an owned string visit.
    fn visit_borrowed_str(&self, val: &str) -> Result<Self::Value, DeserializeError> {
        self.visit_string(val.to_owned())
    }

    #[inline]
    /// Called when the deserializer visits an owned string value.
    fn visit_string(&self, _val: String) -> Result<Self::Value, DeserializeError> {
        Err(DeserializeError::UnsupportedType(ValueType::String))
    }

    #[inline]
    /// Called when the deserializer visits a u64 value.
    fn visit_u64(&self, _val: u64) -> Result<Self::Value, DeserializeError> {
        Err(DeserializeError::UnsupportedType(ValueType::U64))
    }

    #[inline]
    /// Called when the deserializer visits a i64 value.
    fn visit_i64(&self, _val: i64) -> Result<Self::Value, DeserializeError> {
        Err(DeserializeError::UnsupportedType(ValueType::I64))
    }

    #[inline]
    /// Called when the deserializer visits a f64 value.
    fn visit_f64(&self, _val: f64) -> Result<Self::Value, DeserializeError> {
        Err(DeserializeError::UnsupportedType(ValueType::F64))
    }

    #[inline]
    /// Called when the deserializer visits a bool value.
    fn visit_bool(&self, _val: bool) -> Result<Self::Value, DeserializeError> {
        Err(DeserializeError::UnsupportedType(ValueType::Bool))
    }

    #[inline]
    /// Called when the deserializer visits a datetime value.
    fn visit_datetime(&self, _val: DateTime) -> Result<Self::Value, DeserializeError> {
        Err(DeserializeError::UnsupportedType(ValueType::DateTime))
    }

    #[inline]
    /// Called when the deserializer visits an IP address value.
    fn visit_ip_address(&self, _val: Ipv6Addr) -> Result<Self::Value, DeserializeError> {
        Err(DeserializeError::UnsupportedType(ValueType::IpAddr))
    }

    #[inline]
    /// Called when the deserializer visits a facet value.
    fn visit_facet(&self, _val: Facet) -> Result<Self::Value, DeserializeError> {
        Err(DeserializeError::UnsupportedType(ValueType::Facet))
    }

    #[inline]
    /// Called when the deserializer visits a bytes value.
    fn visit_bytes(&self, _val: Vec<u8>) -> Result<Self::Value, DeserializeError> {
        Err(DeserializeError::UnsupportedType(ValueType::Bytes))
    }

    #[inline]
    /// Called when the deserializer visits a pre-tokenized string value.
    fn visit_pre_tokenized_string(
        &self,
        _val: PreTokenizedString,
    ) -> Result<Self::Value, DeserializeError> {
        Err(DeserializeError::UnsupportedType(ValueType::PreTokStr))
    }

    #[inline]
    /// Called when the deserializer visits an array.
    fn visit_array<'de, A>(&self, _access: A) -> Result<Self::Value, DeserializeError>
    where A: ArrayAccess<'de> {
        Err(DeserializeError::UnsupportedType(ValueType::Array))
    }

    #[inline]
    /// Called when the deserializer visits a object value.
    fn visit_object<'de, A>(&self, _access: A) -> Result<Self::Value, DeserializeError>
    where A: ObjectAccess<'de> {
        Err(DeserializeError::UnsupportedType(ValueType::Object))
    }
}

/// Access to a sequence of values which can be deserialized.
pub trait ArrayAccess<'de> {
    /// A indicator as to how many values are in the object.
    ///
    /// This can be used to pre-allocate entries but should not
    /// be depended on as a fixed size.
    fn size_hint(&self) -> usize;

    /// Attempts to deserialize the next element in the sequence.
    fn next_element<V: ValueDeserialize>(&mut self) -> Result<Option<V>, DeserializeError>;

    /// Deserializes the next element using caller-provided state.
    fn next_element_seed<S: ValueDeserializeSeed>(
        &mut self,
        _seed: S,
    ) -> Result<Option<S::Value>, DeserializeError> {
        Err(DeserializeError::custom(
            "Array deserializer does not support seeds",
        ))
    }
}

/// TODO: Improve docs
pub trait ObjectAccess<'de> {
    /// A indicator as to how many values are in the object.
    ///
    /// This can be used to pre-allocate entries but should not
    /// be depended on as a fixed size.
    fn size_hint(&self) -> usize;

    /// Attempts to deserialize the next key-value pair in the object.
    fn next_entry<V: ValueDeserialize>(&mut self) -> Result<Option<(String, V)>, DeserializeError>;

    /// Deserializes the next entry using separate key and value seeds.
    fn next_entry_seed<K: ValueDeserializeSeed, V: ValueDeserializeSeed>(
        &mut self,
        _key: K,
        _value: V,
    ) -> Result<Option<(K::Value, V::Value)>, DeserializeError> {
        Err(DeserializeError::custom(
            "Object deserializer does not support seeds",
        ))
    }
}

/// Controls string ownership and allocation hints during binary deserialization.
pub trait StringRead<R: Read> {
    /// Reads a string from the input.
    fn read_string(reader: &mut R) -> Result<Cow<'_, str>, DeserializeError>;

    /// Bounds a container's allocation hint for the input.
    fn size_hint(reader: &R, length: usize) -> usize;
}

/// Reads owned strings from an arbitrary reader.
pub struct OwnedStrings;

impl<R: Read> StringRead<R> for OwnedStrings {
    fn read_string(reader: &mut R) -> Result<Cow<'_, str>, DeserializeError> {
        Ok(Cow::Owned(<String as BinarySerializable>::deserialize(
            reader,
        )?))
    }

    fn size_hint(_reader: &R, length: usize) -> usize {
        length
    }
}

/// Borrows strings from an input slice.
pub struct BorrowedStrings;

impl StringRead<&[u8]> for BorrowedStrings {
    fn read_string<'a>(reader: &'a mut &[u8]) -> Result<Cow<'a, str>, DeserializeError> {
        let len =
            usize::try_from(VInt::deserialize(reader)?.val()).map_err(DeserializeError::custom)?;
        let text = reader.get(..len).ok_or_else(|| {
            io::Error::new(io::ErrorKind::UnexpectedEof, "Truncated stored string")
        })?;
        let text = std::str::from_utf8(text).map_err(DeserializeError::custom)?;
        *reader = &reader[len..];
        Ok(Cow::Borrowed(text))
    }

    fn size_hint(reader: &&[u8], length: usize) -> usize {
        length.min(reader.len())
    }
}

/// The document deserializer used to read the tantivy documents serialized with
/// `BinarySerializable`.
///
/// This acts very similarly to serde's deserialize types and can incrementally
/// deserialize each field of the document from the provided reader (`R`).
///
/// TODO: Switch to slice instead?
pub struct BinaryDocumentDeserializer<'de, R, M = OwnedStrings> {
    length: usize,
    position: usize,
    reader: &'de mut R,
    mode: PhantomData<M>,
}

impl<'de, R: Read> BinaryDocumentDeserializer<'de, R> {
    pub(crate) fn from_reader(reader: &'de mut R) -> Result<Self, DeserializeError> {
        Self::new(reader)
    }
}

impl<'de, R, M> BinaryDocumentDeserializer<'de, R, M>
where
    R: Read,
    M: StringRead<R>,
{
    fn new(reader: &'de mut R) -> Result<Self, DeserializeError> {
        let length = VInt::deserialize(reader)?;

        Ok(Self {
            length: length.val() as usize,
            position: 0,
            reader,
            mode: PhantomData,
        })
    }

    /// Returns true if the deserializer has deserialized all the entries
    /// within the document.
    fn is_complete(&self) -> bool {
        self.position >= self.length
    }
}

impl<'a, 'de: 'a> BinaryDocumentDeserializer<'a, &'de [u8], BorrowedStrings> {
    /// Creates a deserializer that borrows strings from the remaining input slice.
    pub fn from_slice(reader: &'a mut &'de [u8]) -> Result<Self, DeserializeError> {
        Self::new(reader)
    }
}

impl<'de, R, M> DocumentDeserializer<'de> for BinaryDocumentDeserializer<'de, R, M>
where
    R: Read,
    M: StringRead<R>,
{
    #[inline]
    fn size_hint(&self) -> usize {
        M::size_hint(self.reader, self.length)
    }

    fn next_field<V: ValueDeserialize>(&mut self) -> Result<Option<(Field, V)>, DeserializeError> {
        self.next_field_seed(PhantomData::<V>)
    }

    fn next_field_seed<S: ValueDeserializeSeed>(
        &mut self,
        seed: S,
    ) -> Result<Option<(Field, S::Value)>, DeserializeError> {
        if self.is_complete() {
            return Ok(None);
        }

        let field = Field::deserialize(self.reader).map_err(DeserializeError::from)?;

        let deserializer = BinaryValueDeserializer::<_, M>::from_reader(self.reader)?;
        let value = seed.deserialize(deserializer)?;

        self.position += 1;

        Ok(Some((field, value)))
    }
}

/// A single value deserializer that deserializes a value serialized with `BinarySerializable`.
/// TODO: Improve docs
pub struct BinaryValueDeserializer<'de, R, M = OwnedStrings> {
    value_type: ValueType,
    reader: &'de mut R,
    mode: PhantomData<M>,
}

impl<'de, R, M> BinaryValueDeserializer<'de, R, M>
where
    R: Read,
    M: StringRead<R>,
{
    fn from_reader(reader: &'de mut R) -> Result<Self, DeserializeError> {
        let type_code = <u8 as BinarySerializable>::deserialize(reader)?;

        let value_type = match type_code {
            type_codes::TEXT_CODE => ValueType::String,
            type_codes::U64_CODE => ValueType::U64,
            type_codes::I64_CODE => ValueType::I64,
            type_codes::F64_CODE => ValueType::F64,
            type_codes::BOOL_CODE => ValueType::Bool,
            type_codes::DATE_CODE => ValueType::DateTime,
            type_codes::HIERARCHICAL_FACET_CODE => ValueType::Facet,
            type_codes::BYTES_CODE => ValueType::Bytes,
            type_codes::EXT_CODE => {
                let ext_type_code = <u8 as BinarySerializable>::deserialize(reader)?;

                match ext_type_code {
                    type_codes::TOK_STR_EXT_CODE => ValueType::PreTokStr,
                    _ => {
                        return Err(DeserializeError::from(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "No extended field type is associated with code {ext_type_code:?}"
                            ),
                        )))
                    }
                }
            }
            type_codes::IP_CODE => ValueType::IpAddr,
            type_codes::NULL_CODE => ValueType::Null,
            type_codes::ARRAY_CODE => ValueType::Array,
            type_codes::OBJECT_CODE => ValueType::Object,
            #[allow(deprecated)]
            type_codes::JSON_OBJ_CODE => ValueType::JSONObject,
            _ => {
                return Err(DeserializeError::from(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("No field type is associated with code {type_code:?}"),
                )))
            }
        };

        Ok(Self {
            value_type,
            reader,
            mode: PhantomData,
        })
    }

    fn validate_type(&self, expected_type: ValueType) -> Result<(), DeserializeError> {
        if self.value_type == expected_type {
            Ok(())
        } else {
            Err(DeserializeError::TypeMismatch {
                expected: expected_type,
                actual: self.value_type,
            })
        }
    }
}

impl<'a, 'de: 'a> BinaryValueDeserializer<'a, &'de [u8], BorrowedStrings> {
    /// Creates a deserializer that borrows strings from the remaining input slice.
    pub fn from_slice(reader: &'a mut &'de [u8]) -> Result<Self, DeserializeError> {
        Self::from_reader(reader)
    }
}

impl<'de, R, M> ValueDeserializer<'de> for BinaryValueDeserializer<'de, R, M>
where
    R: Read,
    M: StringRead<R>,
{
    fn deserialize_null(self) -> Result<(), DeserializeError> {
        self.validate_type(ValueType::Null)?;
        Ok(())
    }

    fn deserialize_string(self) -> Result<String, DeserializeError> {
        self.validate_type(ValueType::String)?;
        M::read_string(self.reader).map(Cow::into_owned)
    }

    fn deserialize_u64(self) -> Result<u64, DeserializeError> {
        self.validate_type(ValueType::U64)?;
        <u64 as BinarySerializable>::deserialize(self.reader).map_err(DeserializeError::from)
    }

    fn deserialize_i64(self) -> Result<i64, DeserializeError> {
        self.validate_type(ValueType::I64)?;
        <i64 as BinarySerializable>::deserialize(self.reader).map_err(DeserializeError::from)
    }

    fn deserialize_f64(self) -> Result<f64, DeserializeError> {
        self.validate_type(ValueType::F64)?;
        <u64 as BinarySerializable>::deserialize(self.reader)
            .map(u64_to_f64)
            .map_err(DeserializeError::from)
    }

    fn deserialize_datetime(self) -> Result<DateTime, DeserializeError> {
        self.validate_type(ValueType::DateTime)?;
        <DateTime as BinarySerializable>::deserialize(self.reader).map_err(DeserializeError::from)
    }

    fn deserialize_facet(self) -> Result<Facet, DeserializeError> {
        self.validate_type(ValueType::Facet)?;
        <Facet as BinarySerializable>::deserialize(self.reader).map_err(DeserializeError::from)
    }

    fn deserialize_bytes(self) -> Result<Vec<u8>, DeserializeError> {
        self.validate_type(ValueType::Bytes)?;
        <Vec<u8> as BinarySerializable>::deserialize(self.reader).map_err(DeserializeError::from)
    }

    fn deserialize_ip_address(self) -> Result<Ipv6Addr, DeserializeError> {
        self.validate_type(ValueType::IpAddr)?;
        <u128 as BinarySerializable>::deserialize(self.reader)
            .map(Ipv6Addr::from_u128)
            .map_err(DeserializeError::from)
    }

    fn deserialize_bool(self) -> Result<bool, DeserializeError> {
        self.validate_type(ValueType::Bool)?;
        <bool as BinarySerializable>::deserialize(self.reader).map_err(DeserializeError::from)
    }

    fn deserialize_pre_tokenized_string(self) -> Result<PreTokenizedString, DeserializeError> {
        self.validate_type(ValueType::PreTokStr)?;
        <PreTokenizedString as BinarySerializable>::deserialize(self.reader)
            .map_err(DeserializeError::from)
    }

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, DeserializeError>
    where V: ValueVisitor {
        match self.value_type {
            ValueType::Null => visitor.visit_null(),
            ValueType::String => match M::read_string(self.reader)? {
                Cow::Borrowed(val) => visitor.visit_borrowed_str(val),
                Cow::Owned(val) => visitor.visit_string(val),
            },
            ValueType::U64 => {
                let val = self.deserialize_u64()?;
                visitor.visit_u64(val)
            }
            ValueType::I64 => {
                let val = self.deserialize_i64()?;
                visitor.visit_i64(val)
            }
            ValueType::F64 => {
                let val = self.deserialize_f64()?;
                visitor.visit_f64(val)
            }
            ValueType::DateTime => {
                let val = self.deserialize_datetime()?;
                visitor.visit_datetime(val)
            }
            ValueType::Facet => {
                let val = self.deserialize_facet()?;
                visitor.visit_facet(val)
            }
            ValueType::Bytes => {
                let val = self.deserialize_bytes()?;
                visitor.visit_bytes(val)
            }
            ValueType::IpAddr => {
                let val = self.deserialize_ip_address()?;
                visitor.visit_ip_address(val)
            }
            ValueType::Bool => {
                let val = self.deserialize_bool()?;
                visitor.visit_bool(val)
            }
            ValueType::PreTokStr => {
                let val = self.deserialize_pre_tokenized_string()?;
                visitor.visit_pre_tokenized_string(val)
            }
            ValueType::Array => {
                let access = BinaryArrayDeserializer::<_, M>::from_reader(self.reader)?;
                visitor.visit_array(access)
            }
            ValueType::Object => {
                let access = BinaryObjectDeserializer::<_, M>::from_reader(self.reader)?;
                visitor.visit_object(access)
            }
            #[allow(deprecated)]
            ValueType::JSONObject => {
                let mut de = serde_json::Deserializer::from_reader(self.reader);
                let json_map = <serde_json::Map<String, serde_json::Value> as serde::Deserialize>::deserialize(&mut de)
                    .map_err(DeserializeError::custom)?;
                OwnedValue::from(serde_json::Value::Object(json_map)).deserialize_any(visitor)
            }
        }
    }
}

/// A deserializer for an array of values serialized with `BinarySerializable`.
/// TODO: Improve docs
pub struct BinaryArrayDeserializer<'de, R, M = OwnedStrings> {
    length: usize,
    position: usize,
    reader: &'de mut R,
    mode: PhantomData<M>,
}

impl<'de, R, M> BinaryArrayDeserializer<'de, R, M>
where
    R: Read,
    M: StringRead<R>,
{
    /// Attempts to create a new array deserializer from a given reader.
    fn from_reader(reader: &'de mut R) -> Result<Self, DeserializeError> {
        let length = <VInt as BinarySerializable>::deserialize(reader)?;

        Ok(Self {
            length: length.val() as usize,
            position: 0,
            reader,
            mode: PhantomData,
        })
    }

    /// Returns true if the deserializer has deserialized all the elements
    /// within the array.
    fn is_complete(&self) -> bool {
        self.position >= self.length
    }
}

impl<'de, R, M> ArrayAccess<'de> for BinaryArrayDeserializer<'de, R, M>
where
    R: Read,
    M: StringRead<R>,
{
    #[inline]
    fn size_hint(&self) -> usize {
        M::size_hint(self.reader, self.length)
    }

    fn next_element<V: ValueDeserialize>(&mut self) -> Result<Option<V>, DeserializeError> {
        self.next_element_seed(PhantomData::<V>)
    }

    fn next_element_seed<S: ValueDeserializeSeed>(
        &mut self,
        seed: S,
    ) -> Result<Option<S::Value>, DeserializeError> {
        if self.is_complete() {
            return Ok(None);
        }

        let deserializer = BinaryValueDeserializer::<_, M>::from_reader(self.reader)?;
        let value = seed.deserialize(deserializer)?;

        // Advance the position cursor.
        self.position += 1;

        Ok(Some(value))
    }
}

/// A deserializer for a object consisting of key-value pairs.
pub struct BinaryObjectDeserializer<'de, R, M = OwnedStrings> {
    /// The inner deserializer.
    ///
    /// Internally an object is just represented by an array
    /// in the format of `[key, value, key, value, key, value]`.
    inner: BinaryArrayDeserializer<'de, R, M>,
}

impl<'de, R, M> BinaryObjectDeserializer<'de, R, M>
where
    R: Read,
    M: StringRead<R>,
{
    /// Attempts to create a new object deserializer from a given reader.
    fn from_reader(reader: &'de mut R) -> Result<Self, DeserializeError> {
        let inner = BinaryArrayDeserializer::<_, M>::from_reader(reader)?;
        Ok(Self { inner })
    }
}

impl<'de, R, M> ObjectAccess<'de> for BinaryObjectDeserializer<'de, R, M>
where
    R: Read,
    M: StringRead<R>,
{
    #[inline]
    /// A indicator as to how many values are in the object.
    ///
    /// This can be used to pre-allocate entries but should not
    /// be depended on as a fixed size.
    fn size_hint(&self) -> usize {
        // We divide by 2 here as we know our elements are going to be
        // in the format of `[key, value, key, value, key, value]`.
        self.inner.size_hint() / 2
    }

    /// Attempts to deserialize the next key-value pair in the object.
    fn next_entry<V: ValueDeserialize>(&mut self) -> Result<Option<(String, V)>, DeserializeError> {
        self.next_entry_seed(PhantomData::<String>, PhantomData::<V>)
    }

    fn next_entry_seed<K: ValueDeserializeSeed, V: ValueDeserializeSeed>(
        &mut self,
        key: K,
        value: V,
    ) -> Result<Option<(K::Value, V::Value)>, DeserializeError> {
        if self.inner.is_complete() {
            return Ok(None);
        }
        let key = self
            .inner
            .next_element_seed(key)?
            .ok_or_else(|| DeserializeError::custom("Missing object key"))?;
        let value = self
            .inner
            .next_element_seed(value)?
            .ok_or_else(|| DeserializeError::custom("Missing object value"))?;
        Ok(Some((key, value)))
    }
}

fn owned_value_type(value: &OwnedValue) -> ValueType {
    match value {
        OwnedValue::Null => ValueType::Null,
        OwnedValue::Str(_) => ValueType::String,
        OwnedValue::U64(_) => ValueType::U64,
        OwnedValue::I64(_) => ValueType::I64,
        OwnedValue::F64(_) => ValueType::F64,
        OwnedValue::Date(_) => ValueType::DateTime,
        OwnedValue::Facet(_) => ValueType::Facet,
        OwnedValue::Bytes(_) => ValueType::Bytes,
        OwnedValue::IpAddr(_) => ValueType::IpAddr,
        OwnedValue::Bool(_) => ValueType::Bool,
        OwnedValue::PreTokStr(_) => ValueType::PreTokStr,
        OwnedValue::Array(_) => ValueType::Array,
        OwnedValue::Object(_) => ValueType::Object,
    }
}

macro_rules! deserialize_owned_scalar {
    ($method:ident, $variant:ident, $ty:ty, $kind:ident) => {
        fn $method(self) -> Result<$ty, DeserializeError> {
            match self {
                OwnedValue::$variant(value) => Ok(value),
                value => Err(DeserializeError::TypeMismatch {
                    expected: ValueType::$kind,
                    actual: owned_value_type(&value),
                }),
            }
        }
    };
}

impl<'de> ValueDeserializer<'de> for OwnedValue {
    fn deserialize_null(self) -> Result<(), DeserializeError> {
        match self {
            OwnedValue::Null => Ok(()),
            value => Err(DeserializeError::TypeMismatch {
                expected: ValueType::Null,
                actual: owned_value_type(&value),
            }),
        }
    }

    deserialize_owned_scalar!(deserialize_string, Str, String, String);
    deserialize_owned_scalar!(deserialize_u64, U64, u64, U64);
    deserialize_owned_scalar!(deserialize_i64, I64, i64, I64);
    deserialize_owned_scalar!(deserialize_f64, F64, f64, F64);
    deserialize_owned_scalar!(deserialize_datetime, Date, DateTime, DateTime);
    deserialize_owned_scalar!(deserialize_facet, Facet, Facet, Facet);
    deserialize_owned_scalar!(deserialize_bytes, Bytes, Vec<u8>, Bytes);
    deserialize_owned_scalar!(deserialize_ip_address, IpAddr, Ipv6Addr, IpAddr);
    deserialize_owned_scalar!(deserialize_bool, Bool, bool, Bool);
    deserialize_owned_scalar!(
        deserialize_pre_tokenized_string,
        PreTokStr,
        PreTokenizedString,
        PreTokStr
    );

    fn deserialize_any<V: ValueVisitor>(self, visitor: V) -> Result<V::Value, DeserializeError> {
        match self {
            OwnedValue::Null => visitor.visit_null(),
            OwnedValue::Str(value) => visitor.visit_string(value),
            OwnedValue::U64(value) => visitor.visit_u64(value),
            OwnedValue::I64(value) => visitor.visit_i64(value),
            OwnedValue::F64(value) => visitor.visit_f64(value),
            OwnedValue::Date(value) => visitor.visit_datetime(value),
            OwnedValue::Facet(value) => visitor.visit_facet(value),
            OwnedValue::Bytes(value) => visitor.visit_bytes(value),
            OwnedValue::IpAddr(value) => visitor.visit_ip_address(value),
            OwnedValue::Bool(value) => visitor.visit_bool(value),
            OwnedValue::PreTokStr(value) => visitor.visit_pre_tokenized_string(value),
            OwnedValue::Array(values) => visitor.visit_array(values.into_iter()),
            OwnedValue::Object(values) => visitor.visit_object(values.into_iter()),
        }
    }
}

impl<'de> ArrayAccess<'de> for std::vec::IntoIter<OwnedValue> {
    fn size_hint(&self) -> usize {
        self.len()
    }

    fn next_element<V: ValueDeserialize>(&mut self) -> Result<Option<V>, DeserializeError> {
        self.next_element_seed(PhantomData::<V>)
    }

    fn next_element_seed<S: ValueDeserializeSeed>(
        &mut self,
        seed: S,
    ) -> Result<Option<S::Value>, DeserializeError> {
        self.next().map(|value| seed.deserialize(value)).transpose()
    }
}

impl<'de> ObjectAccess<'de> for std::collections::btree_map::IntoIter<String, OwnedValue> {
    fn size_hint(&self) -> usize {
        self.len()
    }

    fn next_entry<V: ValueDeserialize>(&mut self) -> Result<Option<(String, V)>, DeserializeError> {
        self.next_entry_seed(PhantomData::<String>, PhantomData::<V>)
    }

    fn next_entry_seed<K: ValueDeserializeSeed, V: ValueDeserializeSeed>(
        &mut self,
        key_seed: K,
        value_seed: V,
    ) -> Result<Option<(K::Value, V::Value)>, DeserializeError> {
        self.next()
            .map(|(key, value)| {
                Ok((
                    key_seed.deserialize(OwnedValue::Str(key))?,
                    value_seed.deserialize(value)?,
                ))
            })
            .transpose()
    }
}

// Core type implementations

impl ValueDeserialize for String {
    #[inline]
    fn deserialize<'de, D>(deserializer: D) -> Result<Self, DeserializeError>
    where D: ValueDeserializer<'de> {
        deserializer.deserialize_string()
    }
}

impl ValueDeserialize for u64 {
    #[inline]
    fn deserialize<'de, D>(deserializer: D) -> Result<Self, DeserializeError>
    where D: ValueDeserializer<'de> {
        deserializer.deserialize_u64()
    }
}

impl ValueDeserialize for i64 {
    #[inline]
    fn deserialize<'de, D>(deserializer: D) -> Result<Self, DeserializeError>
    where D: ValueDeserializer<'de> {
        deserializer.deserialize_i64()
    }
}

impl ValueDeserialize for f64 {
    #[inline]
    fn deserialize<'de, D>(deserializer: D) -> Result<Self, DeserializeError>
    where D: ValueDeserializer<'de> {
        deserializer.deserialize_f64()
    }
}

impl ValueDeserialize for DateTime {
    #[inline]
    fn deserialize<'de, D>(deserializer: D) -> Result<Self, DeserializeError>
    where D: ValueDeserializer<'de> {
        deserializer.deserialize_datetime()
    }
}

impl ValueDeserialize for Ipv6Addr {
    #[inline]
    fn deserialize<'de, D>(deserializer: D) -> Result<Self, DeserializeError>
    where D: ValueDeserializer<'de> {
        deserializer.deserialize_ip_address()
    }
}

impl ValueDeserialize for Facet {
    #[inline]
    fn deserialize<'de, D>(deserializer: D) -> Result<Self, DeserializeError>
    where D: ValueDeserializer<'de> {
        deserializer.deserialize_facet()
    }
}

impl ValueDeserialize for Vec<u8> {
    #[inline]
    fn deserialize<'de, D>(deserializer: D) -> Result<Self, DeserializeError>
    where D: ValueDeserializer<'de> {
        deserializer.deserialize_bytes()
    }
}

impl ValueDeserialize for PreTokenizedString {
    #[inline]
    fn deserialize<'de, D>(deserializer: D) -> Result<Self, DeserializeError>
    where D: ValueDeserializer<'de> {
        deserializer.deserialize_pre_tokenized_string()
    }
}

// Collections kind of suck, but can't think of a nicer way of doing this generically
// without quite literally cloning serde entirely...

struct VecVisitor<T: ValueDeserialize>(PhantomData<T>);
impl<T: ValueDeserialize> ValueVisitor for VecVisitor<T> {
    type Value = Vec<T>;

    fn visit_array<'de, A>(&self, mut access: A) -> Result<Self::Value, DeserializeError>
    where A: ArrayAccess<'de> {
        let mut entries = Vec::with_capacity(access.size_hint());
        while let Some(value) = access.next_element()? {
            entries.push(value);
        }
        Ok(entries)
    }
}
impl<T: ValueDeserialize> ValueDeserialize for Vec<T> {
    #[inline]
    fn deserialize<'de, D>(deserializer: D) -> Result<Self, DeserializeError>
    where D: ValueDeserializer<'de> {
        deserializer.deserialize_any(VecVisitor(PhantomData))
    }
}

struct BTreeMapVisitor<T: ValueDeserialize>(PhantomData<T>);
impl<T: ValueDeserialize> ValueVisitor for BTreeMapVisitor<T> {
    type Value = BTreeMap<String, T>;

    fn visit_object<'de, A>(&self, mut access: A) -> Result<Self::Value, DeserializeError>
    where A: ObjectAccess<'de> {
        let mut entries = BTreeMap::new();
        while let Some((key, value)) = access.next_entry()? {
            entries.insert(key, value);
        }
        Ok(entries)
    }
}
impl<T: ValueDeserialize> ValueDeserialize for BTreeMap<String, T> {
    #[inline]
    fn deserialize<'de, D>(deserializer: D) -> Result<Self, DeserializeError>
    where D: ValueDeserializer<'de> {
        deserializer.deserialize_any(BTreeMapVisitor(PhantomData))
    }
}

struct HashMapVisitor<T: ValueDeserialize>(PhantomData<T>);
impl<T: ValueDeserialize> ValueVisitor for HashMapVisitor<T> {
    type Value = HashMap<String, T>;

    fn visit_object<'de, A>(&self, mut access: A) -> Result<Self::Value, DeserializeError>
    where A: ObjectAccess<'de> {
        let mut entries = HashMap::with_capacity(access.size_hint());
        while let Some((key, value)) = access.next_entry()? {
            entries.insert(key, value);
        }
        Ok(entries)
    }
}
impl<T: ValueDeserialize> ValueDeserialize for HashMap<String, T> {
    #[inline]
    fn deserialize<'de, D>(deserializer: D) -> Result<Self, DeserializeError>
    where D: ValueDeserializer<'de> {
        deserializer.deserialize_any(HashMapVisitor(PhantomData))
    }
}

struct KeyValuesVecVisitor<T: ValueDeserialize>(PhantomData<T>);
impl<T: ValueDeserialize> ValueVisitor for KeyValuesVecVisitor<T> {
    type Value = Vec<(String, T)>;

    fn visit_object<'de, A>(&self, mut access: A) -> Result<Self::Value, DeserializeError>
    where A: ObjectAccess<'de> {
        let mut entries = Vec::with_capacity(access.size_hint());
        while let Some(entry) = access.next_entry()? {
            entries.push(entry);
        }
        Ok(entries)
    }
}
impl<T: ValueDeserialize> ValueDeserialize for Vec<(String, T)> {
    #[inline]
    fn deserialize<'de, D>(deserializer: D) -> Result<Self, DeserializeError>
    where D: ValueDeserializer<'de> {
        deserializer.deserialize_any(KeyValuesVecVisitor(PhantomData))
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::io::Cursor;

    use serde_json::Number;
    use tokenizer_api::Token;

    use super::*;
    use crate::schema::document::existing_type_impls::JsonObjectIter;
    use crate::schema::document::se::BinaryValueSerializer;
    use crate::schema::document::{ReferenceValue, ReferenceValueLeaf};

    fn serialize_value<'a>(value: ReferenceValue<'a, &'a serde_json::Value>) -> Vec<u8> {
        let mut writer = Vec::new();

        let mut serializer = BinaryValueSerializer::new(&mut writer);
        serializer.serialize_value(value).expect("Serialize value");

        writer
    }

    fn deserialize_value(buffer: Vec<u8>) -> crate::schema::OwnedValue {
        let mut slice = buffer.as_slice();
        let deserializer = BinaryValueDeserializer::from_slice(&mut slice).unwrap();
        let value = <OwnedValue as ValueDeserialize>::deserialize(deserializer).unwrap();
        assert!(slice.is_empty());
        let mut cursor = Cursor::new(buffer);
        let deserializer = BinaryValueDeserializer::<_>::from_reader(&mut cursor).unwrap();
        let owned = <OwnedValue as ValueDeserialize>::deserialize(deserializer).unwrap();
        assert_eq!(value, owned);
        value
    }

    struct BorrowedStrings<'a> {
        input: &'a [u8],
        strings: RefCell<Vec<String>>,
    }

    impl ValueDeserializeSeed for &BorrowedStrings<'_> {
        type Value = ();

        fn deserialize<'de, D: ValueDeserializer<'de>>(
            self,
            deserializer: D,
        ) -> Result<(), DeserializeError> {
            deserializer.deserialize_any(self)
        }
    }

    impl ValueVisitor for &BorrowedStrings<'_> {
        type Value = ();

        fn visit_borrowed_str(&self, value: &str) -> Result<(), DeserializeError> {
            let start = self.input.as_ptr() as usize;
            let end = start + self.input.len();
            assert!((start..=end).contains(&(value.as_ptr() as usize)));
            assert!(value.as_ptr() as usize + value.len() <= end);
            self.strings.borrow_mut().push(value.to_owned());
            Ok(())
        }

        fn visit_array<'de, A: ArrayAccess<'de>>(
            &self,
            mut access: A,
        ) -> Result<(), DeserializeError> {
            assert!(access.size_hint() <= self.input.len());
            while access.next_element_seed(*self)?.is_some() {}
            Ok(())
        }

        fn visit_object<'de, A: ObjectAccess<'de>>(
            &self,
            mut access: A,
        ) -> Result<(), DeserializeError> {
            assert!(access.size_hint() <= self.input.len());
            while access.next_entry_seed(*self, *self)?.is_some() {}
            Ok(())
        }
    }

    #[test]
    fn test_slice_document_borrows_nested_strings_and_keys() {
        let mut document = crate::TantivyDocument::new();
        let mut schema = crate::schema::Schema::builder();
        let field = schema.add_json_field("payload", crate::schema::STORED);
        let schema = schema.build();
        document.add_text(field, "hello");
        document.add_field_value(field, serde_json::json!({"key": ["nested", ""]}));
        let mut bytes = Vec::new();
        super::super::se::BinaryDocumentSerializer::new(&mut bytes, &schema)
            .serialize_doc(&document)
            .unwrap();
        let visitor = BorrowedStrings {
            input: &bytes,
            strings: RefCell::new(Vec::new()),
        };
        let mut input = bytes.as_slice();
        let mut deserializer = BinaryDocumentDeserializer::from_slice(&mut input).unwrap();
        assert_eq!(
            deserializer.next_field_seed(&visitor).unwrap(),
            Some((field, ()))
        );
        assert_eq!(
            deserializer.next_field_seed(&visitor).unwrap(),
            Some((field, ()))
        );
        assert_eq!(deserializer.next_field_seed(&visitor).unwrap(), None);
        assert!(input.is_empty());
        assert_eq!(*visitor.strings.borrow(), ["hello", "key", "nested", ""]);

        let mut input = bytes.as_slice();
        let decoded = <crate::TantivyDocument as DocumentDeserialize>::deserialize(
            BinaryDocumentDeserializer::from_slice(&mut input).unwrap(),
        )
        .unwrap();
        assert_eq!(decoded, document);
    }

    #[test]
    fn test_slice_legacy_json_normalizes_dates() {
        let json = serde_json::json!({
            "date": "2024-02-03T12:34:56.123456789+05:30",
            "nested": ["2024-02-03T07:04:56Z", "not a date"],
        });
        let mut bytes = vec![type_codes::JSON_OBJ_CODE];
        serde_json::to_writer(&mut bytes, &json).unwrap();
        let decoded = deserialize_value(bytes);
        assert_eq!(decoded, OwnedValue::from(json));
        let OwnedValue::Object(object) = decoded else {
            panic!("object")
        };
        assert!(matches!(object.get("date"), Some(OwnedValue::Date(_))));
    }

    #[test]
    fn test_slice_bounds_document_and_nested_container_hints() {
        let mut bytes = Vec::new();
        VInt(u64::MAX).serialize(&mut bytes).unwrap();
        let mut input = bytes.as_slice();
        let mut deserializer = BinaryDocumentDeserializer::from_slice(&mut input).unwrap();
        assert_eq!(deserializer.size_hint(), 0);
        assert!(deserializer.next_field::<OwnedValue>().is_err());

        for container in [type_codes::ARRAY_CODE, type_codes::OBJECT_CODE] {
            let mut bytes = vec![type_codes::ARRAY_CODE];
            VInt(1).serialize(&mut bytes).unwrap();
            bytes.push(container);
            VInt(u64::MAX).serialize(&mut bytes).unwrap();
            let visitor = BorrowedStrings {
                input: &bytes,
                strings: RefCell::new(Vec::new()),
            };
            let mut input = bytes.as_slice();
            let deserializer = BinaryValueDeserializer::from_slice(&mut input).unwrap();
            assert!(deserializer.deserialize_any(&visitor).is_err());
        }
    }

    #[test]
    fn test_slice_rejects_truncated_and_invalid_strings() {
        let mut oversized_array = vec![type_codes::ARRAY_CODE];
        VInt(u64::MAX).serialize(&mut oversized_array).unwrap();
        for bytes in [
            oversized_array,
            vec![type_codes::TEXT_CODE, 131, b'a'],
            vec![type_codes::TEXT_CODE, 129, 255],
            vec![255],
            vec![type_codes::OBJECT_CODE, 129, type_codes::TEXT_CODE, 128],
        ] {
            let result = BinaryValueDeserializer::from_slice(&mut bytes.as_slice())
                .and_then(<OwnedValue as ValueDeserialize>::deserialize);
            assert!(result.is_err(), "{bytes:?}");
        }
    }

    #[test]
    fn test_simple_value_serialize() {
        let result = serialize_value(ReferenceValueLeaf::Null.into());
        let value = deserialize_value(result);
        assert_eq!(value, crate::schema::OwnedValue::Null);

        let result = serialize_value(ReferenceValueLeaf::Str("hello, world").into());
        let value = deserialize_value(result);
        assert_eq!(
            value,
            crate::schema::OwnedValue::Str(String::from("hello, world"))
        );

        let result = serialize_value(ReferenceValueLeaf::U64(123).into());
        let value = deserialize_value(result);
        assert_eq!(value, crate::schema::OwnedValue::U64(123));

        let result = serialize_value(ReferenceValueLeaf::I64(-123).into());
        let value = deserialize_value(result);
        assert_eq!(value, crate::schema::OwnedValue::I64(-123));

        let result = serialize_value(ReferenceValueLeaf::F64(123.3845).into());
        let value = deserialize_value(result);
        assert_eq!(value, crate::schema::OwnedValue::F64(123.3845));

        let result = serialize_value(ReferenceValueLeaf::Bool(false).into());
        let value = deserialize_value(result);
        assert_eq!(value, crate::schema::OwnedValue::Bool(false));

        let result =
            serialize_value(ReferenceValueLeaf::Date(DateTime::from_timestamp_micros(100)).into());
        let value = deserialize_value(result);
        assert_eq!(
            value,
            crate::schema::OwnedValue::Date(DateTime::from_timestamp_micros(100))
        );

        let facet = Facet::from_text("/hello/world").unwrap();
        let result = serialize_value(ReferenceValueLeaf::Facet(&facet).into());
        let value = deserialize_value(result);
        assert_eq!(value, crate::schema::OwnedValue::Facet(facet));

        let pre_tok_str = PreTokenizedString {
            text: "hello, world".to_string(),
            tokens: vec![Token::default(), Token::default()],
        };
        let result = serialize_value(ReferenceValueLeaf::PreTokStr(&pre_tok_str).into());
        let value = deserialize_value(result);
        assert_eq!(value, crate::schema::OwnedValue::PreTokStr(pre_tok_str));
    }

    #[test]
    fn test_array_serialize() {
        let elements = [serde_json::Value::Null, serde_json::Value::Null];
        let result = serialize_value(ReferenceValue::Array(elements.iter()));
        let value = deserialize_value(result);
        assert_eq!(
            value,
            crate::schema::OwnedValue::Array(vec![
                crate::schema::OwnedValue::Null,
                crate::schema::OwnedValue::Null,
            ]),
        );

        let elements = [
            serde_json::Value::String("Hello, world".into()),
            serde_json::Value::String("Some demo".into()),
        ];
        let result = serialize_value(ReferenceValue::Array(elements.iter()));
        let value = deserialize_value(result);
        assert_eq!(
            value,
            crate::schema::OwnedValue::Array(vec![
                crate::schema::OwnedValue::Str(String::from("Hello, world")),
                crate::schema::OwnedValue::Str(String::from("Some demo")),
            ]),
        );

        let elements = [];
        let result = serialize_value(ReferenceValue::Array(elements.iter()));
        let value = deserialize_value(result);
        assert_eq!(value, crate::schema::OwnedValue::Array(vec![]));

        let elements = [
            serde_json::Value::Null,
            serde_json::Value::String("Hello, world".into()),
            serde_json::Value::Number(12345.into()),
        ];
        let result = serialize_value(ReferenceValue::Array(elements.iter()));
        let value = deserialize_value(result);
        assert_eq!(
            value,
            crate::schema::OwnedValue::Array(vec![
                crate::schema::OwnedValue::Null,
                crate::schema::OwnedValue::Str(String::from("Hello, world")),
                crate::schema::OwnedValue::I64(12345),
            ]),
        );
    }

    #[test]
    fn test_object_serialize() {
        let mut object = serde_json::Map::new();
        object.insert(
            "my-first-key".into(),
            serde_json::Value::String("Hello".into()),
        );
        object.insert("my-second-key".into(), serde_json::Value::Null);
        object.insert(
            "my-third-key".into(),
            serde_json::Value::Number(Number::from_f64(123.0).unwrap()),
        );
        let result = serialize_value(ReferenceValue::Object(JsonObjectIter(object.iter())));
        let value = deserialize_value(result);

        let mut expected_object = BTreeMap::new();
        expected_object.insert(
            "my-first-key".to_string(),
            crate::schema::OwnedValue::Str(String::from("Hello")),
        );
        expected_object.insert("my-second-key".to_string(), crate::schema::OwnedValue::Null);
        expected_object.insert(
            "my-third-key".to_string(),
            crate::schema::OwnedValue::F64(123.0),
        );
        assert_eq!(value, crate::schema::OwnedValue::Object(expected_object));

        let object = serde_json::Map::new();
        let result = serialize_value(ReferenceValue::Object(JsonObjectIter(object.iter())));
        let value = deserialize_value(result);
        let expected_object = BTreeMap::new();
        assert_eq!(value, crate::schema::OwnedValue::Object(expected_object));

        let mut object = serde_json::Map::new();
        object.insert("my-first-key".into(), serde_json::Value::Null);
        object.insert("my-second-key".into(), serde_json::Value::Null);
        object.insert("my-third-key".into(), serde_json::Value::Null);
        let result = serialize_value(ReferenceValue::Object(JsonObjectIter(object.iter())));
        let value = deserialize_value(result);
        let mut expected_object = BTreeMap::new();
        expected_object.insert("my-first-key".to_string(), crate::schema::OwnedValue::Null);
        expected_object.insert("my-second-key".to_string(), crate::schema::OwnedValue::Null);
        expected_object.insert("my-third-key".to_string(), crate::schema::OwnedValue::Null);
        assert_eq!(value, crate::schema::OwnedValue::Object(expected_object));
    }

    #[test]
    fn test_json_compat() {
        let data = [
            8, 123, 34, 107, 101, 121, 97, 58, 34, 58, 34, 98, 108, 117, 98, 34, 44, 34, 118, 97,
            108, 115, 34, 58, 123, 34, 104, 101, 121, 34, 58, 34, 104, 111, 34, 125, 125,
        ]
        .to_vec();
        let expected = json!({
            "keya:": "blub",
            "vals": {
                "hey": "ho"
            }
        });
        let expected_val: OwnedValue = expected.clone().into();

        let value = deserialize_value(data);
        assert_eq!(value, expected_val);
    }

    #[test]
    fn test_nested_serialize() {
        let mut object = serde_json::Map::new();
        object.insert(
            "my-array".into(),
            serde_json::Value::Array(vec![
                serde_json::Value::Null,
                serde_json::Value::String(String::from("bobby of the sea")),
            ]),
        );
        object.insert(
            "my-object".into(),
            serde_json::Value::Object(
                vec![
                    (
                        "inner-1".to_string(),
                        serde_json::Value::Number((-123i64).into()),
                    ),
                    (
                        "inner-2".to_string(),
                        serde_json::Value::String(String::from("bobby of the sea 2")),
                    ),
                ]
                .into_iter()
                .collect(),
            ),
        );
        let result = serialize_value(ReferenceValue::Object(JsonObjectIter(object.iter())));
        let value = deserialize_value(result);

        let mut expected_object = BTreeMap::new();
        expected_object.insert(
            "my-array".to_string(),
            crate::schema::OwnedValue::Array(vec![
                crate::schema::OwnedValue::Null,
                crate::schema::OwnedValue::Str(String::from("bobby of the sea")),
            ]),
        );
        expected_object.insert(
            "my-object".to_string(),
            crate::schema::OwnedValue::Object(
                vec![
                    (
                        "inner-1".to_string(),
                        crate::schema::OwnedValue::I64(-123i64),
                    ),
                    (
                        "inner-2".to_string(),
                        crate::schema::OwnedValue::Str(String::from("bobby of the sea 2")),
                    ),
                ]
                .into_iter()
                .collect(),
            ),
        );
        assert_eq!(value, crate::schema::OwnedValue::Object(expected_object));

        // Some more extreme nesting that might behave weirdly
        let mut object = serde_json::Map::new();
        object.insert(
            "my-array".into(),
            serde_json::Value::Array(vec![serde_json::Value::Array(vec![
                serde_json::Value::Array(vec![]),
                serde_json::Value::Array(vec![serde_json::Value::Null]),
            ])]),
        );
        let result = serialize_value(ReferenceValue::Object(JsonObjectIter(object.iter())));
        let value = deserialize_value(result);

        let mut expected_object = BTreeMap::new();
        expected_object.insert(
            "my-array".to_string(),
            OwnedValue::Array(vec![OwnedValue::Array(vec![
                OwnedValue::Array(vec![]),
                OwnedValue::Array(vec![OwnedValue::Null]),
            ])]),
        );
        assert_eq!(value, OwnedValue::Object(expected_object));
    }
}
