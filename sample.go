package main

import (
	"encoding/csv"
	"log"
	"os"
	"strconv"
)

type Sample struct {
	Date  string
	Value float64
}

func LoadSamples() []Sample {

	file, err := os.Open("clean_weather.csv")

	var samples []Sample

	if err != nil {
		log.Fatal("Error while reading the file", err)
	}

	defer file.Close()

	reader := csv.NewReader(file)

	records, err := reader.ReadAll()

	if err != nil {
		log.Fatal("Error while reading records", err)
	}

	for _, eachrecord := range records[1:900] {

		date := eachrecord[0]
		value := eachrecord[1]

		result, err := strconv.ParseFloat(value, 64)
		if err != nil {
			log.Fatal("Something went wrong.", err)
		}

		samples = append(samples, Sample{
			Date:  date,
			Value: result,
		})
	}

	return samples[1:]
}
