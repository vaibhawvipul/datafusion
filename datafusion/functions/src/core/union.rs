// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::any::Any;

use arrow::{array::ArrayRef, compute::concat, datatypes::DataType};
use datafusion_common::{exec_err, Result};
use datafusion_expr::{ColumnarValue, ScalarUDFImpl, Signature, Volatility};

#[derive(Debug)]
pub struct UnionFunc {
    signature: Signature,
}

impl UnionFunc {
    pub fn new() -> Self {
        Self {
            signature: Signature::variadic_equal(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for UnionFunc {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "union"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {

        if arg_types.len() < 2 {
            return exec_err!("Union function requires at least two arguments");
        }

        let reference_type = &arg_types[0];

        for arg_type in arg_types.iter().skip(1) {
            if arg_type != reference_type {
                return exec_err!("Inconsistent data types in arguments");
            }
        }

        Ok(reference_type.clone())
    }

    fn invoke(&self, args: &[ColumnarValue]) -> Result<ColumnarValue> {

        if args.len() < 2 {
            return exec_err!("Union function requires at least two arguments");
        }

        let arrays: Vec<ArrayRef> = args.iter()
            .map(|arg| match arg {
                ColumnarValue::Array(array) => Ok(array.clone()),
                _ => exec_err!("Invalid argument type for union function"),
            })
            .collect::<Result<Vec<ArrayRef>>>()?;

        // Type checks
        let data_type = arrays[0].data_type().clone();
        for array in &arrays {
            if array.data_type() != &data_type {
                return exec_err!("All arguments to UNION must be of the same type");
            }
        }

        let array_refs: Vec<&dyn arrow::array::Array> = arrays.iter().map(|array| array.as_ref()).collect();

        Ok(ColumnarValue::Array(concat(&array_refs)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, Float64Array, Int64Array};
    use arrow::datatypes::DataType;

    #[test]
    fn test_union_return_type() {
        let union_func = UnionFunc::new();

        // Test with the same type
        let arg_types = vec![DataType::Int64, DataType::Int64];
        assert_eq!(
            union_func.return_type(&arg_types).unwrap(),
            DataType::Int64
        );

        // Test with different types
        let arg_types = vec![DataType::Int64, DataType::Float64];
        assert!(union_func.return_type(&arg_types).is_err());
    }

    #[test]
    fn test_union_invoke() {
        let union_func = UnionFunc::new();

        // Test with two integer arrays
        let array1 = std::sync::Arc::new(Int64Array::from(vec![Some(1), Some(2)])) as ArrayRef;
        let array2 = std::sync::Arc::new(Int64Array::from(vec![Some(3), Some(4)])) as ArrayRef;
        let args = vec![
            ColumnarValue::Array(array1.clone()),
            ColumnarValue::Array(array2.clone()),
        ];

        let result = union_func.invoke(&args).unwrap();
        if let ColumnarValue::Array(result_array) = result {
            let result_array = result_array.as_any().downcast_ref::<Int64Array>().unwrap();
            assert_eq!(result_array.len(), 4);
            assert_eq!(result_array.value(0), 1);
            assert_eq!(result_array.value(1), 2);
            assert_eq!(result_array.value(2), 3);
            assert_eq!(result_array.value(3), 4);
        } else {
            panic!("Unexpected result type");
        }

        // Test with an integer array and a float array
        let array3 = std::sync::Arc::new(Float64Array::from(vec![Some(1.1), Some(2.2)])) as ArrayRef;
        let args = vec![
            ColumnarValue::Array(array1),
            ColumnarValue::Array(array3),
        ];

        assert!(union_func.invoke(&args).is_err());
    }
}
