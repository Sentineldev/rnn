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

func ScaleMatrix(a float64, b *mat.Dense) *mat.Dense {

	matrix := mat.NewDense(b.RawMatrix().Rows, b.RawMatrix().Cols, nil)

	matrix.Scale(a, b)

	return matrix
}

func PowMatrix(a *mat.Dense, b int) *mat.Dense {

	a.Pow(a, b)
	return a

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
	// return XavierInitialization2(NodesOut)
	// return XavierInitialization(NodesIn, NodesOut)
	// x := rand.Float64()
	// x -= 0.1
	// return x
}

func ApplyTanh(i, j int, v float64) float64 {
	return math.Tanh(v)
}

func ApplyDerivTanh(i, j int, v float64) float64 {
	return 1 - math.Pow(math.Tanh(v), 2)
}

func ApplyZeros(i, j int, v float64) float64 {
	return 0
}

func MultVec(a *mat.VecDense, b *mat.VecDense) *mat.VecDense {

	vec := mat.NewVecDense(a.Len(), nil)

	vec.MulElemVec(a, b)

	return vec
}
