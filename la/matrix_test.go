package la_test

import (
	. "github.com/hayden-erickson/neural-network/la"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Matrix", func() {
	var m Matrix
	var data [][]float64

	BeforeEach(func() {
		data = [][]float64{
			{0.0, 4.0, 0.0},
			{1.0, 2.0, 5.0},
			{2.0, 0.0, 0.0},
		}
		m = NewMatrix(data)
	})

	Describe("#At", func() {
		It("returns the correct element", func() {
			Expect(data[1][2]).To(Equal(*m.At(1, 2)))
			Expect(data[0][1]).To(Equal(*m.At(0, 1)))
			Expect(data[2][0]).To(Equal(*m.At(2, 0)))
		})

		It("sets the value when assigned", func() {
			setTo := 4.0
			Expect(*m.At(0, 2)).ToNot(Equal(setTo))
			*m.At(0, 2) = setTo
			Expect(*m.At(0, 2)).To(Equal(setTo))
		})
	})

	Describe("#T", func() {
		It("correctly applies the transpose", func() {
			trans := m.T()
			Expect(trans.At(1, 1)).To(Equal(m.At(1, 1)))
			Expect(trans.At(2, 1)).To(Equal(m.At(1, 2)))
			Expect(trans.At(0, 2)).To(Equal(m.At(2, 0)))

			rm := RandMatrix(5, 10)
			rm.T()
		})
	})

	Describe("#Outer", func() {
		It("returns the outer product", func() {
			a := []float64{1.0, 2.0, 3.0}
			b := []float64{4.0, 5.0}

			c := NewMatrix([][]float64{
				{4.0, 5.0},
				{8.0, 10.0},
				{12.0, 15.0},
			})

			Expect(Outer(a, b)).To(Equal(c))
		})
	})

	Describe("#Dot", func() {
		It("returns the dot product of two vectors", func() {
			a := []float64{1.0, 2.0}
			b := []float64{3.0, 4.0}

			Expect(Dot(a, b)).To(Equal(11.0))

		})
	})

	Describe("#MVDot", func() {
		It("returns the dot product", func() {
			a := NewMatrix([][]float64{
				{1.0, 2.0, 3.0},
				{4.0, 5.0, 5.0},
			})

			b := []float64{1.0, 2.0, 3.0}

			Expect(MVDot(a, b)).To(Equal([]float64{14.0, 29.0}))
		})
	})

	Describe("#MMDOT", func() {
		It("returns the product of two matricies", func() {
			a := NewMatrix([][]float64{
				{1, 2, 3},
				{4, 5, 6},
			})

			b := NewMatrix([][]float64{
				{1, 2, 3, 4},
				{5, 6, 7, 8},
				{9, 10, 11, 12},
			})

			c := MMDot(a, b)

			Expect(c.Shape()[0]).To(Equal(a.Shape()[0]))
			Expect(c.Shape()[1]).To(Equal(b.Shape()[1]))

			data := [][]float64{
				{38, 44, 50, 56},
				{83, 98, 113, 128},
			}

			for i := 0; i < c.Shape()[0]; i++ {
				Expect(c.Row(i)).To(Equal(data[i]))
			}
		})
	})
})
