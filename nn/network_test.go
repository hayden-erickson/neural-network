package nn_test

import (
	"math/rand"

	"github.com/hayden-erickson/neural-network/la"
	. "github.com/hayden-erickson/neural-network/nn"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Network", func() {
	Describe("#NewNetwork", func() {
		Context("Given 1 or less layers", func() {
			It("returns an error", func() {
				_, err := NewNetwork([]int{})

				Expect(err).To(HaveOccurred())

				_, err = NewNetwork([]int{1})

				Expect(err).To(HaveOccurred())
			})
		})

		Context("Given 2 or more layers", func() {
			It("initializes a random network of the given size", func() {
				layerEx := [][]int{
					{728, 20, 10},
					{15, 20, 100, 2, 34, 1, 6},
					{100, 5},
				}

				for _, layers := range layerEx {
					net, _ := NewNetwork(layers)

					Expect(len(net.Weights)).To(Equal(len(layers) - 1))
					Expect(len(net.Biases)).To(Equal(len(layers) - 1))

					for i := 1; i < len(layers); i++ {
						Expect([]int{net.Weights[i-1].Shape()[0], net.Weights[i-1].Shape()[1]}).To(Equal([]int{layers[i], layers[i-1]}))
						Expect(len(net.Biases[i-1])).To(Equal(layers[i]))
					}
				}

			})
		})
	})

	Describe("#Prop", func() {
		var net Network
		var input, output []float64
		var inputSize, outputSize int

		BeforeEach(func() {
			inputSize = rand.Intn(100) + 1
			outputSize = rand.Intn(20) + 1
			net, _ = NewNetwork([]int{inputSize, 10, outputSize})
			input = la.RandVector(inputSize)
		})

		JustBeforeEach(func() {
			sig := Sigmoid
			output = net.Prop(input, sig)
		})

		It("returns the network output", func() {
			Expect(len(output)).To(Equal(outputSize))
		})
	})

	Describe("#BackProp", func() {
		var ex Example
		var net Network
		var inputSize, outputSize int

		BeforeEach(func() {
			inputSize = rand.Intn(50) + 1
			outputSize = rand.Intn(20) + 2
			layers := []int{inputSize, 4, 8, 22, 30, outputSize}
			net, _ = NewNetwork(layers)
		})

		Context("Given the example has a correct size", func() {
			BeforeEach(func() {
				ex = randEx(inputSize, outputSize)
			})

			It("returns the gradients", func() {
				nw, nb := net.BackProp(randEx(inputSize, outputSize), Sigmoid, Quadratic)

				Expect(len(nw)).To(Equal(len(net.Weights)))
				Expect(len(nb)).To(Equal(len(net.Biases)))

				for i, _ := range net.Weights {
					Expect(net.Weights[i].Shape()[0]).To(Equal(nw[i].Shape()[0]))
					Expect(net.Weights[i].Shape()[1]).To(Equal(nw[i].Shape()[1]))
					Expect(len(net.Biases[i])).To(Equal(len(nb[i])))
				}
			})
		})
	})

	Describe("#Saturate", func() {
		var input la.Matrix
		var net Network

		BeforeEach(func() {
			input = la.NewMatrix([][]float64{
				{1, 3, 5, 7},
				{2, 4, 6, 8},
			}, false)

			net, _ = NewNetwork([]int{2, 3, 1})

			net.Weights[0] = la.NewMatrix([][]float64{
				{1, 2},
				{3, 4},
				{5, 6},
			}, false)

			net.Weights[1] = la.NewMatrix([][]float64{{1, 2, 3}}, false)

			net.Biases[0] = []float64{10, 5, 1}
			net.Biases[1] = []float64{4}
		})

		It("returns the weighted input and activation output to every layer in the network", func() {
			weighted, activations := net.Saturate(input, aFunc)
			Expect(len(weighted)).To(Equal(2))
			Expect(len(activations)).To(Equal(3))

			// fmt.Println(weighted)
			Expect(weighted[0]).To(Equal(la.NewMatrix([][]float64{
				{15, 21, 27, 33},
				{16, 30, 44, 58},
				{18, 40, 62, 84},
			}, false)))

			Expect(weighted[1]).To(Equal(la.NewMatrix([][]float64{
				{113 / 3.0, 71, 313 / 3.0, 413 / 3.0},
			}, false)))

			Expect(activations[1]).To(Equal(la.NewMatrix([][]float64{
				{15 / 3.0, 21 / 3.0, 27 / 3.0, 33 / 3.0},
				{16 / 3.0, 30 / 3.0, 44 / 3.0, 58 / 3.0},
				{18 / 3.0, 40 / 3.0, 62 / 3.0, 84 / 3.0},
			}, false)))

			Expect(activations[2]).To(Equal(la.NewMatrix([][]float64{
				{113 / 9.0, 71 / 3.0, 313 / 9.0, 413 / 9.0},
			}, false)))
		})
	})

	Describe("#Delta", func() {
		var actual, desired, weighted la.Matrix

		BeforeEach(func() {
			actual = la.NewMatrix([][]float64{
				{10, 10, 10},
				{5, 5, 5},
			}, false)

			desired = la.NewMatrix([][]float64{
				{5, 5, 5},
				{1, 1, 1},
			}, false)

			weighted = la.NewMatrix([][]float64{
				{3, 6, 9},
				{12, 15, 18},
			}, false)
		})

		It("computes the correct values", func() {
			d := Delta(actual, desired, weighted, aFunc, Quadratic)
			// 1, 2, 3
			// 4, 5, 6
			// *******
			// 5, 5, 5
			// 4, 4, 4
			// ------- =
			// 5, 10, 15
			// 16, 20, 24

			Expect(d).To(Equal(la.NewMatrix([][]float64{
				{5, 5, 5},
				{4, 4, 4},
			}, false)))
		})
	})

	Describe("#MBackProp", func() {
		var N int
		var inputs, desired la.Matrix
		var net Network

		BeforeEach(func() {
			inputSize := rand.Intn(100) + 100
			outputSize := rand.Intn(10) + 10
			N = 100

			inputs = la.RandMatrix(inputSize, N)
			desired = la.RandMatrix(outputSize, N)

			net, _ = NewNetwork([]int{inputSize, rand.Intn(10) + 10, outputSize})
		})

		It("returns the gradients as matricies across all inputs", func() {
			nw, nb := net.MBackProp(inputs, desired, Sigmoid, Quadratic)

			for i, w := range nw {
				Expect(w.Shape()).To(Equal(net.Weights[i].Shape()))
			}

			for i, b := range nb {
				Expect(len(b)).To(Equal(len(net.Biases[i])))
			}
		})
	})
})

type testX struct {
	in  int
	out int
}

func (t testX) GetInput() []float64 {
	return la.RandVector(t.in)
}

func (t testX) GetOutput() []float64 {
	return la.RandVector(t.out)
}

func randEx(in, out int) Example {
	return testX{in, out}
}

type act struct{}

func (a act) Fn(z ...float64) float64 {
	return z[0] / 3.0
}

func (a act) Prime(z ...float64) float64 {
	return z[0] / 3.0
}

var aFunc = act{}
