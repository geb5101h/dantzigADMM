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
 * subject to ||X'(y-Xb)||_Inf <= lambda
 * 
 * We solve using the ADMM algorithm
 */
class DantzigSelectorADMM(
    private var convergenceTol: Double,
    private var maxIterations: Int,
    private var regParam: Double,
    private var rho: Double) extends Serializable {

  def this(convergenceTol: Double, maxIterations: Int, regParam: Double) = this(convergenceTol, maxIterations, regParam, 1.0)

  def this() = this(1.0e-5, 100, 1e-3, 1.0)

  def setRegParam(rp: Double) = { regParam = rp }

  def setConvergenceTol(ct: Double) = { convergenceTol = ct }

  def setMaxIterations(mi: Int) = { maxIterations = mi }

  def run(data: RDD[(Double, Vector)]): DantzigSelectorModel = {

    /* 
     * Currently only supports DenseVector
     * TODO: SparseVector support for large d,
     * RowMatrix support for A for large n. As-is,
     * not suitable for very large d as the covariance
     * matrix of the predictors is stored as a local matrix
     */

    val mat = new RowMatrix(data
      .map(x => Vectors.dense(Array(x._1, 1.0) ++ x._2.toArray)))
    val covMat = mat.computeCovariance().toBreeze.toDenseMatrix
    val d = covMat.rows - 1
    val r = covMat(::, 0)
    val A = covMat(1 to d, 1 to d)
    val gamma = eigSym(A).eigenvalues(0)

    var iter = 1
    var tol = Inf
    var alphaOld, betaOld, uOld, alphaNew, betaNew, uNew = BV[Double](Array.fill(d)(0.0))
    while (iter <= maxIterations && tol >= convergenceTol) {
      alphaOld = alphaNew
      betaOld = betaNew
      uOld = uNew
      val aTimesBetaOld = A * betaOld

      alphaNew = DantzigSelector.winterize(alphaOld + r - aTimesBetaOld, regParam)

      betaNew = DantzigSelector.softThreshold(betaOld - A.t * (aTimesBetaOld - uOld + alphaNew - r) / gamma,
        1 / (gamma * rho))

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
    with RegressionModel
    with Serializable {

  protected def predictPoint(
    dataMatrix: Vector,
    weightMatrix: Vector,
    intercept: Double): Double = {
    weightMatrix.toBreeze.dot(dataMatrix.toBreeze) + intercept
  }

}