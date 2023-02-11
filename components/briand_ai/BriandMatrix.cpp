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

#include "BriandMatrix.hxx"

using namespace std;
using namespace Briand;

Matrix::Matrix(const int& rows, const int& cols, const double& initialValue /*= 0.0*/) {
    this->_rows = rows;
    this->_cols = cols;
    this->InstanceMatrix(initialValue);
}

Matrix::Matrix(const std::initializer_list<std::initializer_list<double>>& m) {
    this->_rows = m.size();
    this->_cols = 0;
    
    this->_matrix = new double*[this->_rows];
    unsigned int i = 0;
    unsigned int j;

    for (auto& r : m) {
        if (this->_cols == 0) this->_cols = r.size();
        if (this->_cols != r.size()) throw out_of_range("Matrix cols not uniform in size");
        this->_matrix[i] = new double[this->_cols];
        j = 0;
        for (auto& c : r) this->_matrix[i][j++] = c;
        i++;
    }
}

void Matrix::InstanceMatrix(const double& initialValue /* = 0.0*/) {
    this->_matrix = new double*[this->_rows];
    for (unsigned int i = 0; i < this->_rows; i++) {
        this->_matrix[i] = new double[this->_cols];
        std::fill_n(this->_matrix[i], this->_cols, initialValue);
    }
}

Matrix::~Matrix() {
    for (unsigned int i = 0; i < this->_rows; i++) {
        delete[] this->_matrix[i];
    }

    delete[] this->_matrix;
}

const unsigned int& Matrix::Rows() const {
    return this->_rows;
}

const unsigned int& Matrix::Cols() const {
    return this->_cols;
}

void Matrix::MultiplyScalar(const double& k) {
    for (unsigned int i = 0; i < this->_rows; i++) {
        for (unsigned int j = 0; j < this->_cols; j++) {
            this->_matrix[i][j] = this->_matrix[i][j] * k;
        }
    }
}

unique_ptr<Matrix> Matrix::MultiplyMatrix(const Matrix& other) {
    // Condition: A x B is possible if number of cols in A equals the number of rows in B
    if (other.Rows() != this->Cols()) throw out_of_range("Matrix A(m,n)*B(n,p) failed: n has different value!");

    // A(m,n) * B(n,p) = C(m,p)
    auto result = make_unique<Matrix>(this->_rows, other.Cols(), 0.0); 

    const unsigned int N = other.Rows();

    for (unsigned int i = 0; i < result->Rows(); i++) {
        for (unsigned int j = 0; j < result->Cols(); j++) {
            for (unsigned int k = 0; k < N; k++)
                (*result.get())[i][j] += (*this)[i][k] * other[k][j];
        }
    }

    return std::move(result);
}

unique_ptr<Matrix> Matrix::MultiplyMatrixHadamard(const Matrix& other) {
    // Condition: A x B is possible if number of cols in A equals the number of rows in B
    if (other.Rows() != this->Rows()) throw out_of_range("Matrix A(m,n)*B(m,n) Hadamard failed: m has different value!");
    if (other.Cols() != this->Cols()) throw out_of_range("Matrix A(m,n)*B(m,n) Hadamard failed: n has different value!");

    // A(m,n) * B(m,n) = C(m,n)
    auto result = make_unique<Matrix>(this->_rows, this->_cols, 0.0); 

    const unsigned int N = other.Rows();

    for (unsigned int i = 0; i < result->Rows(); i++) {
        for (unsigned int j = 0; j < result->Cols(); j++) {
            (*result.get())[i][j] += (*this)[i][j] * other[i][j];
        }
    }

    return std::move(result);
}

unique_ptr<vector<double>> Matrix::MultiplyVector(const vector<double>& v) {
    // Condition: A x v is possible if number of cols in A equals the number of components in v
    if (v.size() != this->Cols()) throw out_of_range("Matrix A(m,n)*v(n) failed: n has different value!");

    auto r = make_unique<vector<double>>();

    for (unsigned int i = 0; i < this->Rows(); i++) {
        double ri = 0;
        for (unsigned int j = 0; j < this->Cols(); j++) {
            ri += this->_matrix[i][j] * v[j];
        }
        r->push_back(ri);
    }

    return std::move(r);
}

void Matrix::ApplyFunction(double (*f)(const double& x)) {
    for (unsigned int i = 0; i < this->_rows; i++) {
        for (unsigned int j = 0; j < this->_cols; j++) {
            this->_matrix[i][j] = f(this->_matrix[i][j]);
        }
    }
}

double*& Matrix::operator[](const int& idx) const {
    return this->_matrix[idx];
}

void Matrix::Print() {
    for (unsigned int i = 0; i < this->_rows; i++) {
        printf("|  ");
        for (unsigned int j = 0; j < this->_cols; j++) {
            printf("%.2lf  ", this->_matrix[i][j]);
        }
        printf("|\n");
    }
}


