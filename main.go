package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"

	"github.com/montanaflynn/stats"
	"gonum.org/v1/gonum/mat"
)

const (
	LearningRate = 0.00000001
)

func GenerateRandomNumber() float64 {
	// Crear un nuevo generador de números aleatorios

	// Generar un número float64 aleatorio entre 0 y 1
	result := rand.Float64()

	// Multiplicar el número por 2 y restar 1 para obtener un rango entre -1 y 1
	return result*2 - 1
}

func GenerateRandomNumber2() float64 {
	// Crear un nuevo generador de números aleatorios

	// Generar un número float64 aleatorio entre 0 y 1
	result := rand.Float64()

	// Multiplicar el número por 2 y restar 1 para obtener un rango entre -1 y 1
	return result/5 - .1
}

func Mse_grad(actual *mat.Dense, predicted *mat.Dense) *mat.Dense {

	return SubMatrix(predicted, actual)
}

func Mse(actual *mat.Dense, predicted *mat.Dense) float64 {

	result := SubMatrix(actual, predicted)

	result.Apply(func(i, j int, v float64) float64 {
		return v * v
	}, result)

	transpose := Transpose(result)
	vec := mat.NewVecDense(transpose.RawMatrix().Cols, transpose.RawRowView(0))
	x, err := stats.Mean(vec.RawVector().Data)
	if err != nil {
		log.Fatal("Error when calculating mean")
	}
	return x
}

func Forward(samples []Sample, layers []Layer) ([]*mat.Dense, []*mat.Dense) {

	var hiddens []*mat.Dense
	var outputs []*mat.Dense
	for i := 0; i < len(layers); i++ {
		hidden := mat.NewDense(len(samples), layers[i].InputWeights.RawMatrix().Cols, nil)
		hidden.Apply(ApplyZeros, hidden)

		output := mat.NewDense(len(samples), layers[i].OutputWeights.RawMatrix().Cols, nil)
		output.Apply(ApplyZeros, output)
		for j, sample := range samples {

			x := mat.NewDense(1, layers[i].InputWeights.RawMatrix().Rows, []float64{sample.Value})

			input_x := MultiplyMatrix(x, layers[i].InputWeights)

			previous := hidden.RowView(int(math.Max(float64(j-1), 0)))

			var elements []float64

			for i := 0; i < previous.Len(); i++ {
				elements = append(elements, previous.AtVec(i))
			}
			previousMatrix := mat.NewDense(1, previous.Len(), elements)
			hidden_x := AddMatrix(AddMatrix(input_x, MultiplyMatrix(previousMatrix, layers[i].HiddenWeights)), layers[i].HiddenBias)
			hidden_x.Apply(ApplyTanh, hidden_x)

			var elements2 []float64

			for i := 0; i < hidden_x.RawMatrix().Rows; i++ {
				for j := 0; j < hidden_x.RawMatrix().Cols; j++ {
					elements2 = append(elements2, hidden_x.At(i, j))
				}
			}
			hidden.SetRow(j, elements2)

			//output layer

			output_x := AddMatrix(MultiplyMatrix(hidden_x, layers[i].OutputWeights), layers[i].OutputBias)

			var elements3 []float64
			for i := 0; i < output_x.RawMatrix().Rows; i++ {
				for j := 0; j < output_x.RawMatrix().Cols; j++ {
					elements3 = append(elements3, output_x.At(i, j))
				}
			}
			output.SetRow(j, elements3)
		}

		hiddens = append(hiddens, hidden)
		outputs = append(outputs, output)
	}
	return hiddens, []*mat.Dense{outputs[len(outputs)-1]}
}

func Backwards(net []Layer, samples []Sample, grad *mat.Dense, hiddens []*mat.Dense) []Layer {

	layers := net
	for i := 0; i < len(layers); i++ {
		hidden := hiddens[i]
		// var next_h_grad *mat.Dense
		o_weight_grad := mat.NewDense(layers[i].OutputWeights.RawMatrix().Rows, layers[i].OutputWeights.RawMatrix().Cols, nil)
		o_bias_grad := mat.NewDense(1, 1, nil)
		h_weight_grad := mat.NewDense(layers[i].HiddenWeights.RawMatrix().Rows, layers[i].HiddenWeights.RawMatrix().Cols, nil)
		h_bias_grad := mat.NewDense(layers[i].HiddenBias.RawMatrix().Rows, layers[i].HiddenBias.RawMatrix().Cols, nil)
		i_weight_grad := mat.NewDense(layers[i].InputWeights.RawMatrix().Rows, layers[i].InputWeights.RawMatrix().Cols, nil)
		var next_h_grad *mat.Dense
		for j := len(samples) - 1; j > -1; j-- {
			out_grad := mat.NewDense(1, 1, []float64{grad.At(j, 0)})

			// fmt.Println("O_grad")
			// printMatrix(out_grad)
			tranpose_hd := Transpose(mat.NewDense(1, hidden.RowView(j).Len(), hidden.RawRowView(j)))
			//oweight grad
			o_weight_grad = AddMatrix(o_weight_grad, MultiplyMatrix(tranpose_hd, out_grad))

			o_bias_grad = AddMatrix(o_bias_grad, out_grad)
			//hiiden grad
			h_grad := MultiplyMatrix(out_grad, Transpose(layers[i].OutputWeights))

			if j < len(samples)-1 {

				hh_grad := MultiplyMatrix(next_h_grad, Transpose(layers[i].HiddenWeights))
				h_grad = AddMatrix(h_grad, hh_grad)

			}

			tanh_deriv := mat.NewDense(1, hidden.RowView(j).Len(), hidden.RawRowView(j))

			tanh_deriv.Apply(ApplyDerivTanh, tanh_deriv)

			temp1 := mat.NewVecDense(tanh_deriv.RowView(0).Len(), tanh_deriv.RawRowView(0))
			temp2 := mat.NewVecDense(h_grad.RowView(0).Len(), h_grad.RawRowView(0))

			aux_vec := MultVec(temp1, temp2)
			h_grad = mat.NewDense(1, aux_vec.Len(), aux_vec.RawVector().Data)

			next_h_grad = h_grad

			if j > 1 {
				tranpose_hd := Transpose(mat.NewDense(1, hidden.RowView(j-1).Len(), hidden.RawRowView(j)))

				h_weight_grad = AddMatrix(h_weight_grad, MultiplyMatrix(tranpose_hd, h_grad))
				h_bias_grad = AddMatrix(h_bias_grad, h_grad)
			}

			result := MultiplyMatrix(mat.NewDense(1, 1, []float64{samples[j].Value}), h_grad)

			i_weight_grad = AddMatrix(i_weight_grad, result)
		}

		lr := LearningRate / float64(len(samples))

		layers[i].InputWeights = SubMatrix(layers[i].InputWeights, ScaleMatrix(lr, i_weight_grad))
		layers[i].HiddenWeights = SubMatrix(layers[i].HiddenWeights, ScaleMatrix(lr, h_weight_grad))
		layers[i].HiddenBias = SubMatrix(layers[i].HiddenBias, ScaleMatrix(lr, h_bias_grad))
		layers[i].OutputWeights = SubMatrix(layers[i].OutputWeights, ScaleMatrix(lr, o_weight_grad))
		layers[i].OutputBias = SubMatrix(layers[i].OutputBias, ScaleMatrix(lr, o_bias_grad))
	}

	return net
}
func ScaleData(data []float64) []float64 {

	mean, _ := stats.Mean(data)

	stdDev, _ := stats.StandardDeviation(data)

	scaledData := make([]float64, len(data))

	for i, v := range data {
		scaledData[i] = (v - mean) / stdDev
	}
	return scaledData
}

func SamplesScaled(samples []Sample) []Sample {

	aux := samples
	var auxFloats []float64

	for _, v := range samples {
		auxFloats = append(auxFloats, v.Value)
	}

	auxFloats = ScaleData(auxFloats)

	for i, v := range auxFloats {
		aux[i].Value = v
	}

	return aux

}

func main() {
	samples := LoadSamples()
	data := SamplesScaled(samples) //140 datas,.

	var layer Layer
	var layers []Layer
	layer.New(1, 4, 1)
	layers = append(layers, layer)
	// var expected []float64
	// for _, sample := range normalizedSamples {
	// 	expected = append(expected, sample.NextDay)
	// }

	fmt.Println("Started weights")
	printMatrix(layers[0].InputWeights)

	epochs := 1500
	for epoch := 0; epoch < epochs; epoch++ {
		seq_length := 7
		epoch_loss := 0.00
		for j := 0; j < len(data)-seq_length; j++ {
			seq_x := data[j:(j + seq_length)]
			// for _, v := range seq_x {
			// 	fmt.Printf("%f\n", v.NextDay)
			// }
			seq_y := data[j:(j + seq_length)]
			var seq_y_values []float64
			for _, v := range seq_y {
				seq_y_values = append(seq_y_values, v.NextDay)
			}
			expectedMatrix := mat.NewDense(seq_length, 1, seq_y_values)
			hiddens, output := Forward(seq_x, layers)
			grad := Mse_grad(expectedMatrix, output[0])
			layers = Backwards(layers, seq_x, grad, hiddens)
			epoch_loss += Mse(expectedMatrix, output[0])
		}
		if epoch%50 == 0 {
			fmt.Printf("Epoch: %f\n", epoch_loss)
			fmt.Printf("Epoch: %d train loss %f\n", epoch, (epoch_loss / float64(len(data))))
		}
	}
	fmt.Println("End weights after 100 epochs")
	printMatrix(layers[0].InputWeights)
	printMatrix(layers[0].InputWeights)
	printMatrix(layers[0].InputWeights)
	printMatrix(layers[0].InputWeights)
}
