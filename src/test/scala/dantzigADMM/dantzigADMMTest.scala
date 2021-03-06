package org.apache.spark.mllib.regression

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.scalatest._
import org.apache.spark.mllib.linalg.{ Vectors, Vector, Matrices, Matrix }

class dantzigADMMTest extends FunSuite {

  val dantzig = new DantzigSelectorADMM(
    convergenceTol = 1e-3,
    maxIterations = 100000,
    regParam = 0.0001,
    rho = 1.0)

  val rand = new scala.util.Random()

  val conf = new SparkConf()
    .setAppName("dantzigTest")
    .setMaster("local[2]")
    .set("spark.executor.memory", "1g")
    .set("spark.driver.memory", "1g")

  val sc = new SparkContext(conf)

  val data = sc.parallelize(
    List.fill(10000) {
      (rand.nextDouble(), Vectors.dense(Array.fill(10) { rand.nextDouble() }))
    })

  test("Testing on sample data") {
    val model = dantzig.run(data)
    println("Return intercept:" + model.intercept)
    println("Return weights:" + model.weights.toArray.map(_.toString).reduce(_ + "," + _))
  }
}