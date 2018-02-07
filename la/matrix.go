package la

type Matrix interface {
	T() Matrix
	Row(int) []float64
	Col(int) []float64
	At(i, j int) *float64
	Shape() []int
	Data() []float64
}

type matrix struct {
	data []float64
	x    int
	y    int
}

type transposer struct {
	m matrix
}

func (t transposer) T() Matrix {
	return t.m
}

func (t transposer) Row(i int) []float64 {
	return t.m.Col(i)
}

func (t transposer) Col(j int) []float64 {
	return t.m.Row(j)
}

func (t transposer) At(i, j int) *float64 {
	return t.m.At(j, i)
}

func (t transposer) Shape() []int {
	return []int{t.m.y, t.m.x}
}

func (t transposer) Data() []float64 {
	return t.m.Data()
}

func (m matrix) T() Matrix {
	return transposer{m: m}
}

func (m matrix) Row(i int) []float64 {
	return m.data[(i * m.y):((i + 1) * m.y)]
}

func (m matrix) Col(j int) []float64 {
	out := make([]float64, m.x)

	for i := 0; i < m.x; i++ {
		out[i] = *m.At(i, j)
	}

	return out
}

func (m matrix) At(i, j int) *float64 {
	return &m.data[i*m.y+j]
}

func (m matrix) Shape() []int {
	return []int{m.x, m.y}
}

func (m matrix) Data() []float64 {
	return m.data
}

// func CreateMatrixAgg
