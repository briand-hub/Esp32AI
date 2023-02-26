/** Copyright (C) 2023 briand (https://github.com/briand-hub)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#ifndef BRIAND_MATRIX_H
#define BRIAND_MATRIX_H

#include "BriandInclude.hxx"

using namespace std;

namespace Briand {

    /** @brief Small matrix library. 
        If a more performing way of calculus is found then you need only to change the implementation here!
    */
    class Matrix {
        protected:

        /// @brief Columns
        size_t _cols;

        /// @brief Rows
        size_t _rows;
        
        /// @brief Internal matrix
        double** _matrix;

        /// @brief Instance internal data structures and allocate memory.
        /// @param initialValue initial value of elements
        void InstanceMatrix(const double& initialValue = 0.0);

        public:

        /// @brief Build a new matrix RxC with initial value
        /// @param rows 
        /// @param cols 
        /// @param initialValue initial value for elements (default 0)
        Matrix(const int& rows, const int& cols, const double& initialValue = 0.0);

        /// @brief Build a new matrix RxC with given input initialization matrix
        /// @param m initial values
        Matrix(const std::initializer_list<std::initializer_list<double>>& m);

        /// @brief Useful copy constructor
        Matrix(const Matrix& other);

        ~Matrix();

        /// @brief Return row number
        /// @return rows
        const size_t& Rows() const;

        /// @brief Return col number
        /// @return cols
        const size_t& Cols() const;

        /// @brief Randomize all matrix values
        void Randomize();

        /// @brief Multiply current matrix by a value.
        /// @param k value
        void MultiplyScalar(const double& k);

        /// @brief Multiply current matrix by a vector
        /// @param v vector
        /// @return Pointer to resulting vector
        unique_ptr<vector<double>> MultiplyVector(const vector<double>& v);

        /// @brief Multiply current matrix with other (dot operation). If input matrix is m*n other matrix must be n*p. Result will be a m*p matrix.
        /// @param other Matrix 
        /// @return new matrix
        unique_ptr<Matrix> MultiplyMatrix(const Matrix& other);

        /// @brief Multiply current matrix with other (Hadamard product). 
        /// If input matrix is m*n a(i,j) elements other matrix must be m*n b(i,j) elements. Result will be a m*n matrix where elements are a(i,j)*b(i,j).
        /// @param other Matrix 
        /// @return Matrix result
        unique_ptr<Matrix> MultiplyMatrixHadamard(const Matrix& other);
        
        /// @brief Dot multiplication of two vectors. Assuming vector v2 is transposed.
        /// @param v1 Vector 1
        /// @param v2t Vector 2 (assume transposed)
        /// @return Dot product resulting matrix
        static unique_ptr<Matrix> DotMultiplyVectors(const vector<double>& v1, const vector<double>& v2t);

        /// @brief Apply f() function to all matrix elements
        /// @param f the function to apply f(x)
        unique_ptr<Matrix> ApplyFunction(double (*f)(const double& x));

        /// @brief Transpose operation. If input matrix is m*n a(i,j) returns n*m matrix with a(j,i) elements.
        /// @return Transposed Matrix
        unique_ptr<Matrix> Transpose();

        /// @brief Opertor m[i] returns the internal matrix row
        /// @param idx row index
        /// @return reference to internal pointer
        double*& operator[](const size_t& idx) const;

        /// @brief Reference to element at i,j
        /// @param i row index
        /// @param j column index
        /// @return Element at i,j
        double& at(const size_t& i, const size_t& j);

        /// @brief Print out matrix for debug
        void Print();

        /// @brief Print out a vector for debug
        static void PrintVector(const vector<double>& v);
    };
}

#endif