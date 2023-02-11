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
    if (other.Rows() != this->Cols()) throw runtime_error("Matrix A(m,n)*B(n,p) failed: n has different value!");

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

void Matrix::MultiplyVector(const double*& v, const unsigned int& size) {
    throw runtime_error("Unimplemented");

}

void Matrix::MultiplyVector(const vector<double>& v) {
    throw runtime_error("Unimplemented");
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


