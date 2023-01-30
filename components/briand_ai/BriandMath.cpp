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

#include "BriandMath.hxx"

using namespace std;

double Briand::Math::Identity(const double& x) {
    return x;
}

double Briand::Math::ReLU(const double& x) {
    return x > 0 ? x : 0;
}

double Briand::Math::Sigmoid(const double& x) {
    return 1 / (1 + exp(-1 * x));
}

double Briand::Math::WeightedSum(const vector<double>& values, const vector<double>& weights) {
    // Check vector length is equal
    if (values.size() != weights.size()) throw runtime_error("Briand::ActivationFunctions::WeightedSum - values and weights mismatch size.");

    double sum = 0.0;

    for (uint32_t i = 0; i < values.size(); i++) sum += values.at(i) * weights.at(i);

    return sum; 
}

double Briand::Math::Random() {
    return esp_random() / static_cast<double>(UINT32_MAX);
}

double Briand::Math::MSE(const double& target, const double& output) {
    //return 0.5 * pow(target-output, 2.0);
    return 0.5 * (target - output) * (target - output);
}
