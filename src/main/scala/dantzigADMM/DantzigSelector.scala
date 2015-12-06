package org.apache.spark.mllib.regression

import org.apache.spark.mllib.linalg.{ Vector, Vectors, Matrix }
import breeze.linalg.{ DenseVector => DBV, DenseMatrix => DBM, diag, max, min, norm, eigSym, Vector => BV, Matrix => BM }
import breeze.math._
import breeze.numerics._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary
import org.apache.spark.mllib.linalg.distributed.RowMatrix

/*
 * Implements the Dantzig selector, which is the solution to
 * min{b} ||b||_1 
 * subject to ||X(y-Xb)||_Inf <= lambda
 * 
 * We solve using the ADMM algorithm
 */
class DantzigSelectorADMM(
    private var convergenceTol: Double,
    private var maxIterations: Int,
    private var regParam: Double) extends Serializable {

  def run(data: RDD[(Double, Vector)]): DantzigSelectorModel = {

    //currently only supports DenseVectors
    val mat = new RowMatrix(data
      .map(x => Vectors.dense(Array(x._1, 1.0) ++ x._2.toArray)))
    val covMat = mat.computeCovariance().toBreeze.toDenseMatrix.toDenseMatrix
    val d = covMat.rows - 1
    val r = covMat(::, 0)
    val A = covMat(1 to d, 1 to d)

    var iter = 1
    var tol = Inf
    var alphaOld, betaOld, uOld, alphaNew, betaNew, uNew = BV[Double](Array.fill(d)(0.0))
    while (iter <= maxIterations || tol >= convergenceTol) {
      alphaOld = alphaNew
      betaOld = betaNew
      uOld = uNew
      val aTimesBetaOld = A * betaOld

      alphaNew = DantzigSelector.winterize(alphaOld + r - aTimesBetaOld, regParam)

      betaNew = DantzigSelector.softThreshold(betaOld - A.t * (aTimesBetaOld - uOld + alphaNew - r), 2.0)

      uNew = uOld + r - alphaNew - A * betaNew

      val betaDiff = betaNew - betaOld
      tol = betaDiff dot betaDiff
      iter += 1
    }
    //extract intercept from weights vector
    val betaReturn = Vectors.fromBreeze(betaNew(-0))
    val intReturn = betaNew(0)

    new DantzigSelectorModel(
      betaReturn, intReturn)
  }

}

/*
 * Some utility methods useful
 * in solving the Dantzig selector
 */
object DantzigSelector {
  def winterize(x: BV[Double], lambda: Double): BV[Double] = {
    val len = x.size
    val xNew = x.copy
    for (i <- 0 to len - 1) {
      xNew(i) = signum(xNew(i)) * min(lambda, math.abs(xNew(i)))
    }
    xNew
  }

  def softThreshold(x: BV[Double], lambda: Double): BV[Double] = {
    val len = x.size
    val xNew = x.copy
    for (i <- 0 to len - 1) {
      xNew(i) = signum(xNew(i)) * max(0.0, math.abs(xNew(i) - lambda))
    }
    xNew
  }

}
class DantzigSelectorModel(
  weights: Vector,
  intercept: Double) extends GeneralizedLinearModel(weights, intercept)
    with Serializable {

  protected def predictPoint(
    dataMatrix: Vector,
    weightMatrix: Vector,
    intercept: Double): Double = {
    weightMatrix.toBreeze.dot(dataMatrix.toBreeze) + intercept
  }

}
