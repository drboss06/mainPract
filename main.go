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
	var tps []tp
	for i := 0; i < len(reader[0]); i++ {
		var types tp
		tps = append(tps, types)
	}
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
	var types []string
	for _, tp := range tps {
		if tp.Bool == n {
			types = append(types, "Bool")
		} else if tp.Int == n {
			types = append(types, "Int")
		} else if tp.Datetime == n {
			types = append(types, "Datetime")
		} else {
			types = append(types, "String")
		}
	}
	return types
} // Проверяет на типы первые n строк файла и возвращает массив типов для каждого столбца соответственно

func getSQL(reader [][]string, dbName string, tableName string) (createQuery string, insertQuerys []string) {

	var types = getTypes(reader, 5)

	createQuery = fmt.Sprintf("CREATE TABLE IF NOT EXISTS %s.%s (", dbName, tableName)
	for n, columnName := range reader[0] {
		createQuery += strings.Replace(columnName, " ", "_", -1) + " " + types[n] + ", "
	}
	createQuery = strings.TrimRight(createQuery, ", ")
	createQuery += ") ENGINE = Memory AS SELECT 1;\n"

	var dt = "1/2/2006 3:04:05 PM"

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

	for i := 1; i < len(reader); i += 25000 {
		var query []string
		var n = i + 25000
		if i+25000 >= len(reader) {
			n = len(reader)
		}
		for _, j := range reader[i:n] {
			query = append(query, strings.Join(j, ", "))
		}
		insertQuerys = append(insertQuerys, fmt.Sprintf("INSERT INTO %s.%s (*) VALUES (", dbName, tableName)+strings.Join(query, "), (")+")")
	}
	return createQuery, insertQuerys
} // формирует и возвращает запрос создания таблицы и массив запросов вставки данных чанками по 25000

func timer(name string) func() {
	start := time.Now()
	return func() {
		fmt.Printf("%s took %v\n", name, time.Since(start))
	}
} // Просто таймер, чтобы посмотреть за сколько выполняется парсинг

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

	fmt.Println("Create querys...")

	createQuery, insertQuerys := getSQL(reader, dbName, tableName)
	_, err = conn.Query(ctx, createQuery)

	fmt.Println("Create table...")

	if err != nil && err.Error() != "EOF" && err.Error() != "code: 20, message: Number of columns doesn't match" {
		fmt.Println(createQuery)
		log.Fatal(err.Error())
	}

	fmt.Println("Insert data...")

	for _, i := range insertQuerys {
		if _, err := conn.Query(ctx, i); err != nil && err.Error() != "EOF" {
			log.Fatal(err.Error())
		}
	}

	fmt.Println("All right")
}
