package nn_test

import (
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
					{15, 20, 100},
					{100, 5},
				}

				for _, layers := range layerEx {
					net, _ := NewNetwork(layers)

					Expect(net.NumLayers).To(Equal(len(layers)))
					Expect(len(net.Weights)).To(Equal(len(layers) - 1))
					Expect(len(net.Biases)).To(Equal(len(layers) - 1))

					for i := 1; i < len(layers); i++ {
						Expect(net.Weights[i-1].Shape()).To(BeEquivalentTo([]int{layers[i], layers[i-1]}))
						Expect(net.Biases[i-1].Shape()).To(BeEquivalentTo([]int{layers[i]}))
					}
				}

			})
		})
	})
})
