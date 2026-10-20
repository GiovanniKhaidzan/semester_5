-- SELECT CountryCode, count(District) as totalDistricts
-- FROM city
-- GROUP BY CountryCode

-- SELECT Continent, Region, COUNT(*) as jumlahNegara
-- FROM country
-- GROUP BY Continent, region
-- order by jumlahNegara DESC;


-- SELECT CountryCode, COUNT(language) as totalBahasa
-- from countrylanguage
-- WHERE isOfficial = 'T'
-- GROUP BY CountryCode

-- SELECT region, continent, SUM(population) as totalPopulation
-- From country
-- WHERE Continent = 'Asia'
-- GROUP BY Region
-- HAVING totalPopulation >= 100000000\
