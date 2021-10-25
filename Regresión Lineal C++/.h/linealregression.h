#ifndef LINEALREGRESSION_H
#define LINEALREGRESSION_H

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>

class linealregression
{
public:
    linealregression(){}
    float FuncionCosto(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta);
    std::tuple<Eigen::VectorXd, std::vector<float>> GradienteDescendiente(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta, float alpha, int iteraciones);
};

#endif // LINEALREGRESSION_H
