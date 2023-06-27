package main

import (
	"context"
	"encoding/csv"
	"fmt"
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
	var query_string = "INSERT INTO main_table ("
	var csvMet = CsvProp{}
	//var flag = true
	var flag2 = false
	var counter = 0
	var counter2 = 0
	reader := csvMet.parceCsvFile("./test2.csv")

	for _, i := range reader {
		counter = 0
		for _, j := range reader[0] {
			counter += 1
			if counter == len(i) {
				//flag = false
				flag2 = true
				query_string += j + ") VALUES ("
			} else {
				query_string += j + ","
			}
		}

		if flag2 {
			counter = 0
			for _, j := range i {
				counter += 1
				if counter == len(i) {
					query_string += j + ")"
				} else {
					query_string += j + ","
				}
			}
		}
		flag2 = true
		//if err = conn.Exec(ctx, query_string, false); err != nil {
		//	log.Fatal(err.Error())
		//	break
		//}
		counter2 += 1
		if counter2 == 2 {
			if err = conn.Exec(ctx, query_string, false); err != nil {
				log.Fatal(err.Error())
				break
			}
			break
		}
		query_string = "INSERT INTO main_table ("
	}

	fmt.Print("All right")
}
