package main

import (
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

type Layer struct {
	InputWeights  *mat.Dense
	HiddenWeights *mat.Dense

	HiddenBias    *mat.Dense
	OutputWeights *mat.Dense
	OutputBias    *mat.Dense
}

func (layer *Layer) New(nodesInt int64, nodesOut int64, outputs int64) {

	rand.Seed(time.Now().UnixNano())
	k := 1 / math.Sqrt(float64(nodesOut))

	layer.InputWeights = mat.NewDense(int(nodesInt), int(nodesOut), nil)

	layer.InputWeights.Apply(func(i, j int, v float64) float64 {
		return GenerateRandomNumber()*2*k - k
	}, layer.InputWeights)

	layer.HiddenWeights = mat.NewDense(int(nodesOut), int(nodesOut), nil)
	layer.HiddenWeights.Apply(func(i, j int, v float64) float64 {
		return GenerateRandomNumber()*2*k - k
	}, layer.HiddenWeights)

	layer.HiddenBias = mat.NewDense(1, int(nodesOut), nil)

	layer.HiddenBias.Apply(func(i, j int, v float64) float64 {
		return GenerateRandomNumber()*2*k - k
	}, layer.HiddenBias)

	layer.OutputWeights = mat.NewDense(int(nodesOut), int(outputs), nil)

	layer.OutputWeights.Apply(func(i, j int, v float64) float64 {
		return GenerateRandomNumber()*2*k - k
	}, layer.OutputWeights)
	layer.OutputBias = mat.NewDense(1, int(outputs), nil)

	layer.OutputBias.Apply(func(i, j int, v float64) float64 {
		return GenerateRandomNumber()*2*k - k
	}, layer.OutputBias)
}

func (layer Layer) Matrixes() (*mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense) {

	return layer.InputWeights, layer.HiddenWeights, layer.HiddenBias, layer.OutputWeights, layer.OutputBias
}
