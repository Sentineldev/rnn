package main

import (
	"encoding/csv"
	"log"
	"os"
	"strconv"
)

type Sample struct {
	Date    string
	Value   float64
	Value1  float64
	Value2  float64
	NextDay float64
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

	for _, eachrecord := range records {

		date := eachrecord[0]
		value := eachrecord[1]
		value2 := eachrecord[4]
		value3 := eachrecord[2]
		value4 := eachrecord[3]

		result, err := strconv.ParseFloat(value, 64)
		if err != nil {
			// log.Fatal("Something went wrong.", err)
			result = 0
		}
		result2, err := strconv.ParseFloat(value2, 64)
		if err != nil {
			// log.Fatal("Something went wrong.", err)
			result2 = 0
		}
		result3, err := strconv.ParseFloat(value3, 64)
		if err != nil {
			// log.Fatal("Something went wrong.", err)
			result3 = 0
		}

		result4, err := strconv.ParseFloat(value4, 64)
		if err != nil {
			// log.Fatal("Something went wrong.", err)
			result4 = 0
		}
		samples = append(samples, Sample{
			Date:    date,
			Value:   result,
			Value1:  result3,
			Value2:  result4,
			NextDay: result2,
		})
	}

	return samples[1:]
}
