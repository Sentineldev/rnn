package main

import (
	"gonum.org/v1/gonum/mat"
)

type Layer struct {
	InputWeights  *mat.Dense
	HiddenWeights *mat.Dense

	HiddenBias    *mat.Dense
	OutputWeights *mat.Dense
	OutputBias    *mat.Dense
}

func (layer *Layer) FromArrayMatrixes(data PostPredictRnn) {

	InputWeights := mat.NewDense(len(data.Input_Weights), len(data.Input_Weights[0]), nil)

	for i := 0; i < len(data.Input_Weights); i++ {
		for j := 0; j < len(data.Input_Weights[i]); j++ {
			InputWeights.Set(i, j, data.Input_Weights[i][j])
		}
	}
	layer.InputWeights = InputWeights

	HiddenWeights := mat.NewDense(len(data.Hidden_Weights), len(data.Hidden_Weights[0]), nil)
	for i := 0; i < len(data.Hidden_Weights); i++ {
		for j := 0; j < len(data.Hidden_Weights[i]); j++ {
			HiddenWeights.Set(i, j, data.Hidden_Weights[i][j])
		}
	}

	layer.HiddenWeights = HiddenWeights

	HiddenBias := mat.NewDense(len(data.Hidden_Bias), len(data.Hidden_Bias[0]), nil)

	for i := 0; i < len(data.Hidden_Bias); i++ {
		for j := 0; j < len(data.Hidden_Bias[i]); j++ {
			HiddenBias.Set(i, j, data.Hidden_Bias[i][j])
		}
	}
	layer.HiddenBias = HiddenBias

	OutputWeights := mat.NewDense(len(data.Output_Weights), len(data.Output_Weights[0]), nil)
	for i := 0; i < len(data.Output_Weights); i++ {
		for j := 0; j < len(data.Output_Weights[i]); j++ {
			OutputWeights.Set(i, j, data.Output_Weights[i][j])
		}
	}
	layer.OutputWeights = OutputWeights

	OutputBias := mat.NewDense(len(data.Output_Bias), len(data.Output_Bias[0]), nil)
	for i := 0; i < len(data.Output_Bias); i++ {
		for j := 0; j < len(data.Output_Bias[i]); j++ {
			OutputBias.Set(i, j, data.Output_Bias[i][j])
		}
	}
	layer.OutputBias = OutputBias
}

func (layer *Layer) New(nodesInt int64, nodesOut int64, outputs int64) {

	// k := 1 / math.Sqrt(float64(nodesOut))

	layer.InputWeights = mat.NewDense(int(nodesInt), int(nodesOut), nil)

	// layer.InputWeights.Apply(func(i, j int, v float64) float64 {
	// 	return GenerateRandomNumber()*2*k - k
	// }, layer.InputWeights)
	layer.InputWeights.Apply(ApplyRandomNumbers2, layer.InputWeights)

	layer.HiddenWeights = mat.NewDense(int(nodesOut), int(nodesOut), nil)
	// layer.HiddenWeights.Apply(func(i, j int, v float64) float64 {
	// 	return GenerateRandomNumber()*2*k - k
	// }, layer.HiddenWeights)
	layer.HiddenWeights.Apply(ApplyRandomNumbers2, layer.HiddenWeights)

	layer.HiddenBias = mat.NewDense(1, int(nodesOut), nil)

	// layer.HiddenBias.Apply(func(i, j int, v float64) float64 {
	// 	return GenerateRandomNumber()*2*k - k
	// }, layer.HiddenBias)
	layer.HiddenBias.Apply(ApplyRandomNumbers2, layer.HiddenBias)

	layer.OutputWeights = mat.NewDense(int(nodesOut), int(outputs), nil)

	// layer.OutputWeights.Apply(func(i, j int, v float64) float64 {
	// 	return GenerateRandomNumber()*2*k - k
	// }, layer.OutputWeights)

	layer.OutputWeights.Apply(ApplyRandomNumbers2, layer.OutputWeights)

	layer.OutputBias = mat.NewDense(1, int(outputs), nil)
	layer.OutputBias.Apply(ApplyRandomNumbers2, layer.OutputBias)
	// layer.OutputBias.Apply(func(i, j int, v float64) float64 {
	// 	return GenerateRandomNumber()*2*k - k
	// }, layer.OutputBias)
}

type Matrix [][]float64

func (layer Layer) GetArrays() (Matrix, Matrix, Matrix, Matrix, Matrix) {

	var input_weights [][]float64
	var hidden_weights [][]float64
	var hidden_bias [][]float64
	var output_weights [][]float64
	var output_bias [][]float64

	for i := 0; i < layer.InputWeights.RawMatrix().Rows; i++ {
		input_weights = append(input_weights, layer.InputWeights.RawRowView(i))
	}

	for i := 0; i < layer.HiddenWeights.RawMatrix().Rows; i++ {
		hidden_weights = append(hidden_weights, layer.HiddenWeights.RawRowView(i))
	}

	for i := 0; i < layer.HiddenBias.RawMatrix().Rows; i++ {
		hidden_bias = append(hidden_bias, layer.HiddenBias.RawRowView(i))
	}

	for i := 0; i < layer.OutputWeights.RawMatrix().Rows; i++ {
		output_weights = append(output_weights, layer.OutputWeights.RawRowView(i))
	}

	for i := 0; i < layer.OutputBias.RawMatrix().Rows; i++ {
		output_bias = append(output_bias, layer.OutputBias.RawRowView(i))
	}

	return input_weights, hidden_weights, hidden_bias, output_weights, output_bias

}

func (layer Layer) Matrixes() (*mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense) {

	return layer.InputWeights, layer.HiddenWeights, layer.HiddenBias, layer.OutputWeights, layer.OutputBias
}
