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
