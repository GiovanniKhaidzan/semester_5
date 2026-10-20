-- SELECT CountryCode, count(District) as totalDistricts
-- FROM city
-- GROUP BY CountryCode

SELECT Continent, Region, COUNT(*) as jumlahNegara
FROM country
GROUP BY Continent
order by jumlahNegara DESC;
