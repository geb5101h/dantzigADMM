package org.apache.spark.mllib.regression

import org.apache.spark.mllib.linalg.{ Vector, Vectors, Matrix }
import breeze.linalg.{ DenseVector => DBV, DenseMatrix => DBM, diag, max, min, sum, norm, eigSym, Vector => BV, Matrix => BM }
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

    val n = data.count

    val gram = mat.computeGramianMatrix().toBreeze.toDenseMatrix
    val d = gram.rows - 1
    val r = (1.0 / n) * gram(1 to d, 0)
    var A = (1.0 / n) * gram(1 to d, 1 to d)
    A = (A + A.t) / 2.0
    val gamma = breeze.linalg.sum(abs(A * A), breeze.linalg.Axis._0).max
    //val gamma =math.pow(eigSym(A).eigenvalues(d - 1),2)

    var iter = 1
    var tol = Inf
    var alphaOld, betaOld, uOld, alphaNew, betaNew, uNew, aTimesBetaOld = BV[Double](Array.fill(d)(0.0))
    while (iter <= maxIterations && tol >= convergenceTol) {
      var maxDiff = 0.0

      alphaOld = alphaNew
      betaOld = betaNew
      uOld = uNew
      aTimesBetaOld = A * betaOld

      alphaNew = DantzigSelector.winterize(r - aTimesBetaOld - uOld, regParam)

      maxDiff = max(maxDiff, sum(abs(alphaNew.toDenseVector - alphaOld.toDenseVector)))

      betaNew = DantzigSelector.softThreshold(betaOld - A * (aTimesBetaOld + uOld + alphaNew - r) / gamma,
        1 / (gamma * rho))

      maxDiff = max(maxDiff, sum(abs(betaNew.toDenseVector - betaOld.toDenseVector)))

      uNew = uOld + (alphaNew + A * betaNew - r)

      maxDiff = max(maxDiff, sum(abs(uNew.toDenseVector - uOld.toDenseVector)))

      tol = maxDiff

      iter += 1
    }

    println("Finished after " + iter + " iterations\nwith tolerance " + tol)

    //extract intercept from weights vector
    val betaReturn = Vectors.fromBreeze(betaNew(1 to d - 1).toDenseVector)
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
      xNew(i) = signum(xNew(i)) * max(0.0, math.abs(xNew(i)) - lambda)
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
