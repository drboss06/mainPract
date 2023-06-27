package main

import (
	"context"
	"encoding/csv"
	"log"
	"mainPract/repository"
	"os"
)

type CsvProp struct {
}

func (c *CsvProp) readCsvFile(filePath string) [][]string {
	f, err := os.Open(filePath)
	if err != nil {
		log.Fatal("Unable to read input file "+filePath, err)
	}
	defer f.Close()

	csvReader := csv.NewReader(f)
	records, err := csvReader.ReadAll()
	if err != nil {
		log.Fatal("Unable to parse file as CSV for "+filePath, err)
	}

	return records
}

func (c *CsvProp) parceCsvFile(filePath string) [][]string {
	f, err := os.Open(filePath)
	if err != nil {
		log.Fatal("Unable to read input file "+filePath, err)
	}
	defer f.Close()

	csvReader := csv.NewReader(f)

	records, err := csvReader.ReadAll()
	if err != nil {
		log.Fatal("Unable to parse file as CSV for "+filePath, err)
	}

	return records
}

func main() {
	var db = repository.DbClick{}

	conn, err := db.Connect()
	if err != nil {
		panic((err))
	}

	ctx := context.Background()
	//rows, err := conn.Query(ctx, "SELECT t.* FROM default.test_table t LIMIT 501")
	//
	//if err != nil {
	//	log.Panicf(err.Error())
	//}
	//for rows.Next() {
	//	var (
	//		col1 int32
	//	)
	//	if err := rows.Scan(&col1); err != nil {
	//		log.Panicf(err.Error())
	//	}
	//	fmt.Printf("row: col1=%d", col1)
	//}
	var query_string = "INSERT INTO main_table VALUES ("
	var csvMet = CsvProp{}
	var flag = false
	for _, i := range csvMet.parceCsvFile("./test2.csv") {
		if flag {
			for _, j := range i {
				query_string += j + ","
			}
		}
		flag = true
	}
	query_string += ")"
	if err = conn.AsyncInsert(ctx, query_string, false); err != nil {
		log.Fatal(err.Error())
	}
}
