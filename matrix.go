package main

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

func MultiplyMatrix(a *mat.Dense, b *mat.Dense) *mat.Dense {

	matrix := mat.NewDense(a.RawMatrix().Rows, b.RawMatrix().Cols, nil)

	matrix.Mul(a, b)

	return matrix
}

func AddMatrix(a *mat.Dense, b *mat.Dense) *mat.Dense {

	matrix := mat.NewDense(a.RawMatrix().Rows, b.RawMatrix().Cols, nil)

	matrix.Add(a, b)

	return matrix
}

func SubMatrix(a *mat.Dense, b *mat.Dense) *mat.Dense {

	matrix := mat.NewDense(a.RawMatrix().Rows, b.RawMatrix().Cols, nil)

	matrix.Sub(a, b)

	return matrix
}

func Transpose(a *mat.Dense) *mat.Dense {

	transpose := a.T()
	tranposeMatrix := mat.NewDense(1, 1, nil)

	tranposeMatrix.CloneFrom(transpose)

	return tranposeMatrix
}

func printMatrix(a mat.Matrix) {

	fmt.Println(mat.Formatted(a, mat.Prefix("")))
	fmt.Println()
}
func ApplyRandomNumbers(i, j int, v float64) float64 {

	return GenerateRandomNumber()
}

func ApplyRandomNumbers2(i, j int, v float64) float64 {

	return GenerateRandomNumber2()
}

func ApplyTanh(i, j int, v float64) float64 {
	return math.Tanh(v)
}

func ApplyZeros(i, j int, v float64) float64 {
	return 0
}
