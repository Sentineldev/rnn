package main

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
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

func main() {

	samples := LoadSamples()

	var layer Layer
	var layers []Layer

	layer.New(1, 4, 1)
	layers = append(layers, layer)
	hiddens, outputs := Forward(samples[:10], layers)

	fmt.Printf("Hiddens\n")
	for _, x := range hiddens {
		printMatrix(x)
	}
	fmt.Printf("Outputs\n")
	for _, x := range outputs {
		printMatrix(x)
	}

	// rand.Seed(time.Now().Unix())

	// i_weigth := mat.NewDense(1, 5, nil)
	// i_weigth.Apply(ApplyRandomNumbers2, i_weigth)

	// printMatrix(i_weigth)

	// h_weigth := mat.NewDense(5, 5, nil)
	// h_weigth.Apply(ApplyRandomNumbers2, h_weigth)

	// h_bias := mat.NewDense(1, 5, nil)
	// h_bias.Apply(ApplyRandomNumbers2, h_bias)

	// o_weight := mat.NewDense(5, 1, nil)
	// o_weight.Apply(ApplyRandomNumbers, o_weight)

	// o_bias := mat.NewDense(1, 1, nil)
	// o_bias.Apply(ApplyRandomNumbers, o_bias)

	// outputs := []*mat.Dense{}
	// hiddens := [3][]*mat.Dense{}

	// var prev_hidden *mat.Dense
	// for i := 0; i < 3; i++ {

	// 	x := mat.NewDense(1, 1, []float64{samples[i].Value})

	// 	printMatrix(x)

	// 	xi := MultiplyMatrix(x, i_weigth)

	// 	var xh *mat.Dense
	// 	if i == 0 {
	// 		xh = AddMatrix(MultiplyMatrix(xi, h_weigth), h_bias)
	// 	} else {
	// 		xh = AddMatrix(MultiplyMatrix(AddMatrix(xi, prev_hidden), h_weigth), h_bias)
	// 		// xh = AddMatrix(AddMatrix(xi, MultiplyMatrix(prev_hidden, h_weigth)), h_bias)
	// 		// xh = AddMatrix(xi, MultiplyMatrix(prev_hidden, h_weigth))
	// 		// xh = AddMatrix(xh, h_bias)
	// 	}

	// 	xh.Apply(func(i, j int, v float64) float64 {

	// 		return math.Tanh(v)
	// 	}, xh)

	// 	prev_hidden = xh

	// 	printMatrix(xh)

	// 	hiddens[i] = append(hiddens[i], xh)

	// 	xo := AddMatrix(MultiplyMatrix(xh, o_weight), o_bias)

	// 	outputs = append(outputs, xo)
	// }

	// fmt.Printf("Outputs\n")
	// for _, x := range outputs {
	// 	printMatrix(x)

	// }
}