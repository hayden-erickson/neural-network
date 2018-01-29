package la

type Matrix struct {
	X    int
	Y    int
	Data []float64
}

func (m Matrix) At(i, j int) *float64 {
	return &m.Data[i*m.Y+j]
}

func (m Matrix) Row(i int) []float64 {
	return m.Data[(i * m.Y):((i + 1) * m.Y)]
}

func (m Matrix) T() Matrix {
	trans := make([]float64, m.X*m.Y)

	for i := 0; i < m.X; i++ {
		for j := 0; j < m.Y; j++ {
			trans[j*m.X+i] = m.Data[i*m.Y+j]
		}
	}

	return Matrix{
		X:    m.Y,
		Y:    m.X,
		Data: trans,
	}
}

func MVDot(a Matrix, b []float64) []float64 {
	var d []float64

	var from, to int

	for i := 0; i < a.X; i++ {
		from = i * a.Y
		to = (i + 1) * a.Y

		d = append(d, Dot(a.Data[from:to], b))
	}

	return d
}

func MMDot(a, b Matrix) Matrix {
	out := Matrix{
		X:    a.X,
		Y:    b.Y,
		Data: make([]float64, a.X*b.Y),
	}

	bT := b.T()

	for i := 0; i < a.X; i++ {
		for j := 0; j < bT.X; j++ {
			*out.At(i, j) = Dot(a.Data[(i*a.Y):((i+1)*a.Y)], bT.Data[(j*bT.Y):((j+1)*bT.Y)])
		}
	}

	return out
}

func CopyMatrix(into, from Matrix) {
	into.X = from.X
	into.Y = from.Y
	copy(into.Data, from.Data)
}
