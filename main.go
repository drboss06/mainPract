package main

import (
	"context"
	"encoding/csv"
	"fmt"
	"log"
	"mainPract/repository"
	"os"
	"strconv"
	"strings"
	"time"
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

func getTypes(reader [][]string, n int) []string {
	type tp struct {
		Int      int
		Datetime int
		Bool     int
		String   int
	}
	tps := make([]tp, len(reader[0]))
	for index, i := range reader[1:] {
		if index == n {
			break
		}
		var dt = "1/2/2006 3:04:05 PM"
		var noneDate, _ = time.Parse("0000-01-01 00:00:00", "0000-01-01 00:00:00")

		for b, j := range i {
			if _, err := strconv.Atoi(j); err == nil {
				tps[b].Int += 1
			} else if strings.ToLower(j) == "true" || strings.ToLower(j) == "false" {
				tps[b].Bool += 1
			} else if t, err := time.Parse(dt, j); err == nil && t != noneDate {
				tps[b].Datetime += 1
			} else {
				tps[b].String += 1
			}
		}
	}
	types := make([]string, len(tps))
	for i, tp := range tps {
		if tp.Bool == n {
			types[i] = "Bool"
		} else if tp.Int == n {
			types[i] = "Int"
		} else if tp.Datetime == n {
			types[i] = "Datetime"
		} else {
			types[i] = "String"
		}
	}
	return types
}

func getSQL(reader [][]string, dbName string, tableName string) (createQuery string, insertQuerys []string) {
	types := getTypes(reader, 5)

	createQuery = fmt.Sprintf("CREATE TABLE IF NOT EXISTS %s.%s (", dbName, tableName)
	for n, columnName := range reader[0] {
		createQuery += strings.Replace(columnName, " ", "_", -1) + " " + types[n] + ", "
	}
	createQuery = strings.TrimRight(createQuery, ", ")
	createQuery += ") ENGINE = Memory;\n"

	dt := "1/2/2006 3:04:05 PM"

	for _, i := range reader[1:] {
		for n, j := range i {
			switch types[n] {
			case "Int":
				i[n] = j
			case "Datetime":
				if t, err := time.Parse(dt, j); err == nil {
					i[n] = "'" + t.Format("2006-01-02 15:04:05") + "'"
				}
			case "Bool":
				j = strings.ToLower(j)
				if j == "true" {
					i[n] = "1"
				} else {
					i[n] = "0"
				}
			case "String":
				i[n] = "'" + j + "'"
			}
		}
	}

	insertQuerys = make([]string, 0, len(reader)/25000+1)

	for i := 1; i < len(reader); i += 25000 {
		n := i + 25000
		if n > len(reader) {
			n = len(reader)
		}
		query := make([]string, 0, n-i)
		for _, j := range reader[i:n] {
			query = append(query, "("+strings.Join(j, ", ")+")")
		}
		insertQuerys = append(insertQuerys, fmt.Sprintf("INSERT INTO %s.%s (*) VALUES ", dbName, tableName)+
			strings.Join(query, ", "))
	}
	return createQuery, insertQuerys
}

func timer(name string) func() {
	start := time.Now()
	return func() {
		fmt.Printf("%s took %v\n", name, time.Since(start))
	}
}

func main() {
	defer timer("main")()

	var db = repository.DbClick{}

	conn, err := db.Connect()
	if err != nil {
		panic(err)
	}

	ctx := context.Background()

	var csvMet = CsvProp{}
	var reader = csvMet.parceCsvFile("./test2.csv")
	var tableName = "main_table"
	var dbName = "default"

	fmt.Println("Create queries...")

	createQuery, insertQueries := getSQL(reader, dbName, tableName)
	_, err = conn.Query(ctx, createQuery)

	fmt.Println("Create table...")

	if err != nil && err.Error() != "EOF" && err.Error() != "code: 20, message: Number of columns doesn't match" {
		fmt.Println(createQuery)
		log.Fatal(err.Error())
	}

	fmt.Println("Insert data...")

	//for _, query := range insertQueries {
	//	if _, err := conn.Query(ctx, query); err != nil && err.Error() != "EOF" {
	//		log.Fatal(err.Error())
	//	}
	//}

	for _, query := range insertQueries {
		if err := conn.AsyncInsert(ctx, query, false); err != nil && err.Error() != "EOF" {
			log.Fatal(err.Error())
		}
	}

	fmt.Println("All right")
}
