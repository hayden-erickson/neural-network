package loaders

import "github.com/hayden-erickson/neural-network/nn"

type LoaderFunc func(filename string) <-chan nn.Example

func (l LoaderFunc) Load(filename string) <-chan nn.Example {
	return l(filename)
}

type Loader interface {
	Load() ([]nn.Example, error)
}
