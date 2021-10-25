#include "linealregression.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>


/* Se necesita entrenar el modelo, lo que implica alguna funcion de
 * costo y de esta forma se puede medir la precision de la fincion de
 * hipotesis. La funcion de costo es la forma de penalizar al modelo por
 *  cometer un error. */

/* X = variable independiente
 * y = variable dependiente
 * theta = pendiente */

float linealregression::FuncionCosto(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta){
    Eigen::MatrixXd diferencia = pow((X*theta-y).array(),2);
    return diferencia.sum()/(2*X.rows());
}

/* Se implementa la funcion para dar al algoritmo los valores iniciales
 * de theta, que cambiaran iterativamente hasta que converga al valor minimo
 * de la funcion de costo. Basicamente decribira el gradiente descendiente: el
 * es dado por la derivada parcial de la funcion. La funcion tiene un alpha que
 * representa el salto del gradiente y el numero de iteraciones que se necesitan
 * para actualizar theta hasta que la funcion converga al minimo */

std::tuple<Eigen::VectorXd, std::vector<float>> linealregression::GradienteDescendiente(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta, float alpha, int iteraciones){
    /* Almacenamiento temporal para los valores de theta */
    Eigen::MatrixXd temporal = theta;

    /* Variable con la cantidad de parametros m (FEACTURES) */
    int parametros = theta.rows();

    /* Ubicar el costo inicial, que se actualizara iterativamente con los pesos */
    std::vector<float> costo;
    costo.push_back(FuncionCosto(X,y,theta));

    /* Por cada iterecion se calcula la funcion de error */
    for (int i = 0; i < iteraciones; ++i) {
        Eigen::MatrixXd error = X*theta-y;
        for (int j = 0; j < parametros; ++j) {
            Eigen::MatrixXd X_i = X.col(j);
            Eigen::MatrixXd termino = error.cwiseProduct(X_i);
            temporal(j,0) = theta(j,0) - ((alpha/X.rows())*termino.sum());
        }
        theta = temporal;
        costo.push_back(FuncionCosto(X,y,theta));
    }

    return std::make_tuple(theta,costo);
}
