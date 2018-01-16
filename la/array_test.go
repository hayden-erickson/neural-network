package la_test

import (
	. "github.com/hayden-erickson/neural-network/la"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Array", func() {
	Describe("#Vdot", func() {
		var a, b Vector

		Context("Given both vectors are not the same size", func() {
			It("returns an error", func() {
				a = []Scalar{1.0, 2.0}
				b = []Scalar{3.0}

				_, err := Vdot(a, b)

				Expect(err).To(HaveOccurred())
			})
		})

		Context("Given both vectors are the same size", func() {
			It("returns their dot product", func() {
				a = []Scalar{1.0, 2.0, 3.0}
				b = []Scalar{4.0, 5.0, 6.0}

				c, _ := Vdot(a, b)

				Expect(c).To(BeEquivalentTo(32.0))

				a = []Scalar{4.0, 5.0}
				b = []Scalar{3.0, 2.0}

				c, _ = Vdot(a, b)

				Expect(c).To(BeEquivalentTo(22.0))
			})
		})
	})

	Describe("#MVdot", func() {
		var m Matrix
		var v Vector

		Context("Given the inner dimensions dont match", func() {
			It("returns an error", func() {
				m = []Vector{
					{1.0, 2.0, 3.0},
					{4.0, 5.0, 6.0},
				}

				v = []Scalar{1.0, 2.0}

				_, err := MVdot(m, v)

				Expect(err).To(HaveOccurred())
			})
		})

		Context("Given the inner dimensions match", func() {
			It("returns their dot product", func() {

				m = []Vector{
					{1.0, 2.0, 3.0},
					{4.0, 5.0, 6.0},
					{7.0, 8.0, 9.0},
				}

				v = []Scalar{2.0, 3.0, 4.0}

				c, _ := MVdot(m, v)

				Expect(c.Shape()).To(Equal([]int{3}))
				Expect(c).To(BeEquivalentTo([]Scalar{20.0, 47.0, 74.0}))

				m = []Vector{
					{1.0, 2.0},
					{3.0, 4.0},
				}

				v = []Scalar{2.0, 3.0}

				c, _ = MVdot(m, v)

				Expect(c.Shape()).To(Equal([]int{2}))
				Expect(c).To(BeEquivalentTo([]Scalar{8.0, 18.0}))
			})
		})
	})

	Describe("#OP", func() {
		var a, b Vector
		var op SOP

		Context("Given the two vectors are of unequal size", func() {
			It("returns an error", func() {
				a = []Scalar{1.0, 2.0}
				b = []Scalar{1.0, 2.0, 3.0}

				_, err := OP(a, b, op)

				Expect(err).To(HaveOccurred())
			})
		})

		Context("Given the two vectors are of equal size", func() {
			BeforeEach(func() {
				op = func(a, b Scalar) Scalar {
					return a * b
				}
			})

			It("returns their element-wise product", func() {
				a = []Scalar{1.0, 2.0, 3.0}
				b = []Scalar{3.0, 2.0, 1.0}

				c, _ := OP(a, b, op)

				Expect(c).To(BeEquivalentTo([]Scalar{3.0, 4.0, 3.0}))

				a = []Scalar{5.0, 6.0, 7.0}
				b = []Scalar{3.0, 2.0, 1.0}

				c, _ = OP(a, b, op)

				Expect(c).To(BeEquivalentTo([]Scalar{15.0, 12.0, 7.0}))
			})
		})
	})

	Describe("#Map", func() {
		It("applies a function to every element in the vector", func() {
			v := []Scalar{1.0, 2.0, 3.0}

			mapper := func(f Scalar) Scalar {
				return f * 2
			}

			Expect(Map(v, mapper)).To(BeEquivalentTo([]Scalar{2.0, 4.0, 6.0}))

			v = []Scalar{4.0, 5.0}

			Expect(Map(v, mapper)).To(BeEquivalentTo([]Scalar{8.0, 10.0}))
		})
	})
})
