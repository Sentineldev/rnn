package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"

	"github.com/montanaflynn/stats"
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

			x := mat.NewDense(1, layers[i].InputWeights.RawMatrix().Rows, []float64{sample.Value, sample.Value1, sample.Value2})

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

			result := MultiplyMatrix(mat.NewDense(3, 1, []float64{samples[j].Value, samples[j].Value1, samples[j].Value2}), h_grad)

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

func XavierInitialization(nodesIn int64, nodesOut int64) float64 {

	variance := 2.0 / (float64(NodesIn) + float64(nodesOut))

	return rand.Float64() * math.Sqrt(variance)
}

func XavierInitialization2(nodesOut int64) float64 {

	k := 1 / math.Sqrt(NodesOut)
	return GenerateRandomNumber2()*2*k - k
}

func GetArrayPage(array []Sample, currentPage int64, elementsPerPage int64) []Sample {

	indiceInicial := (currentPage - 1) * elementsPerPage

	indiceFinal := indiceInicial + elementsPerPage

	elements := array[indiceInicial:indiceFinal]

	return elements
}

func Train(epochs int, hiddenNeurons int64) []Layer {

	samples := LoadSamples()
	data := SamplesScaled(samples) //140 datas,.

	train_slice := int64(float64(len(data)) * 0.5)
	train_test_slice := int64(float64(len(data)) * 0.4)
	train_data_x := data[:train_slice]
	train_valid := data[len(train_data_x) : len(data)-int(train_test_slice)]
	var layer Layer
	var layers []Layer
	layer.New(3, hiddenNeurons, 1)
	layers = append(layers, layer)

	fmt.Println("Initializing training...")
	for epoch := 0; epoch < epochs; epoch++ {
		seq_length := 7
		epoch_loss := 0.00

		elementsPerPage := train_slice
		pages := len(train_data_x) / int(elementsPerPage)
		var train_data []Sample
		for i := 1; i <= pages; i++ {

			train_data = GetArrayPage(train_data_x, int64(i), int64(elementsPerPage))
			for j := 0; j < len(train_data)-seq_length; j++ {
				seq_x := train_data[j:(j + seq_length)]
				seq_y := train_data[j:(j + seq_length)]
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
		}
		if epoch%10 == 0 {

			valid_loss := 0.00

			for j := 0; j < len(train_valid)-seq_length; j++ {
				seq_x := train_data[j:(j + seq_length)]
				seq_y := train_data[j:(j + seq_length)]
				var seq_y_values []float64
				for _, v := range seq_y {
					seq_y_values = append(seq_y_values, v.NextDay)
				}
				expectedMatrix := mat.NewDense(seq_length, 1, seq_y_values)
				_, y := Forward(seq_x, layers)
				valid_loss += Mse(expectedMatrix, y[0])

				// fmt.Println("Matriz esperada.")
				// printMatrix(expectedMatrix)
				// fmt.Println("Matriz obtenida.")
				// printMatrix(y[0])
			}
			if math.IsNaN(epoch_loss) {
				Train(epochs, hiddenNeurons)
				return layers
			}
			fmt.Printf("Epoch: %f\n", epoch_loss)
			train_loss := (epoch_loss / float64(len(train_data)))
			valid_loss_x := valid_loss / float64(len(train_valid))
			fmt.Printf("Epoch: %d train loss %f valid loss %f\n", epoch, train_loss, valid_loss_x)
		}

	}
	return layers
}
