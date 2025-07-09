-- PART A
create schema mini_project;

use mini_project;
-- 1.	Import the csv file to a table in the database.
-- Right click on schema and import csv via "Table Data Import Wizard". Data is imported as text because it contains some unwanted values i.e. '-'.
-- Replace '-' with Null
	update icccricket set Inn = Null where Inn = '-';
    update icccricket set No = Null where No = '-';
    update icccricket set Runs = Null where Runs = '-';
    update icccricket set HS = Null where HS = '-';
    update icccricket set Avg = Null where Avg = '-';
    update icccricket set `100` = Null where `100` = '-';
    update icccricket set `50` = Null where `50` = '-';
    update icccricket set `0` = Null where `0` = '-';

-- Update column's data types to its correct types    
    alter table icccricket modify column Inn int;
    alter table icccricket modify column No int;
    alter table icccricket modify column Runs int;
--  alter table icccricket modify column HS int;            -- It contains '*' which is an extra information. We will take care of this column later
    alter table icccricket modify column Avg double;
    alter table icccricket modify column `100` int;
    alter table icccricket modify column `50` int;
    alter table icccricket modify column `0` int;

-- 2.	Remove the column 'Player Profile' from the table.
		alter table icccricket drop column `Player Profile`;
        
        
-- 3.	Extract the country name and player names from the given data and store it in separate columns for further usage.
	alter table icccricket add column name text;
    alter table icccricket add column country text;
    
    update icccricket set name = substring(Player,1,locate('(',Player)-2);
    update icccricket set country =  substring(substring_index( substring_index(player,'(',-1),')',1),locate('/', substring_index( substring_index(player,'(',-1),')',1))+1,1000);
    
-- 4.	From the column 'Span' extract the start_year and end_year and store them in separate columns for further usage.
	alter table icccricket add column start_year char(4);
    alter table icccricket add column end_year char(4);
    
	update icccricket set start_year = substr(span,1,4);
    update icccricket set end_year = substr(span,6,9);
    
-- 5.	The column 'HS' has the highest score scored by the player so far in any given match. The column also has details if the player had completed the match in a NOT OUT status. Extract the data and store the highest runs and the NOT OUT status in different columns.
alter table icccricket add column NotOut boolean;
update icccricket set NotOut = case when HS like '%*%' then True else False end;
update icccricket set HS = replace(HS,'*','');
alter table icccricket modify column HS int;


-- 6.	Using the data given, considering the players who were active in the year of 2019, create a set of batting order of best 6 players using the selection criteria of those who have a good average score across all matches for India.
	select * from icccricket where country = 'INDIA' and start_year <= 2019 and end_year >= 2019 order by Avg Desc limit 6;
    
-- 7.	Using the data given, considering the players who were active in the year of 2019, create a set of batting order of best 6 players using the selection criteria of those who have the highest number of 100s across all matches for India.
select * from icccricket where country = 'INDIA' and start_year <= 2019 and end_year >= 2019 order by `100` Desc limit 6;

-- 8.	Using the data given, considering the players who were active in the year of 2019, create a set of batting order of best 6 players using 2 selection criteria of your own for India.
select * from icccricket where country = 'INDIA' and start_year <= 2019 and end_year >= 2019 order by `100` desc, Runs Desc limit 6;

-- 9.	Create a View named ‘Batting_Order_GoodAvgScorers_SA’ using the data given, considering the players who were active in the year of 2019, 
-- create a set of batting order of best 6 players using the selection criteria of those who have a good average score across all matches for South Africa.
create view Batting_Order_GoodAvgScorers_SA as
select * from icccricket 
 where country = 'SA' and start_year <= 2019 and end_year >= 2019 order by Avg Desc limit 6;

-- 10.	Create a View named ‘Batting_Order_HighestCenturyScorers_SA’ Using the data given, considering the players who were active in the year of 2019, 
-- create a set of batting order of best 6 players using the selection criteria of those who have highest number of 100s across all matches for South Africa.
create view Batting_Order_HighestCenturyScorers_SA as
select * from icccricket where country = 'SA' and start_year <= 2019 and end_year >= 2019 order by `100` desc limit 6;

-- 11.	Using the data given, Give the number of player_played for each country.
select country, count(*) as no_of_players from icccricket group by country;

-- 12.	Using the data given, Give the number of player_played for Asian and Non-Asian continent

select case when b.continent = 'Asia' then 'Asian'
else 'Non-Asian' end as continent, count(*) as no_of_players from icccricket as a inner join
( select country,
case
when country in ('India','PAK','SL','BDESH') then 'Asian'
else 'Non-Asian'
end as continent from icccricket group by country ) as b on a.country = b.country
group by b.continent;




-- Part B
use supply_chain;
-- 1.	Company sells the product at different discounted rates. Refer actual product price in product table and selling price in the order item table. 
-- Write a query to find out total amount saved in each order then display the orders from highest to lowest amount saved. 
select orderid, sum((p.unitprice - o.unitprice ) * quantity) as total_amount_saved 
from orderitem as o
inner join product as p on o.productid = p.id
group by orderid;



-- 2.	Mr. Kavin want to become a supplier. He got the database of "Richard's Supply" for reference. Help him to pick: 
-- 		a. List few products that he should choose based on demand.
		select productid, sum(quantity) as quantity_sold from orderitem group by productid order by sum(quantity) desc limit 10;
        
--  	b. Who will be the competitors for him for the products suggested in above questions.
		create temporary table temp select productid, sum(quantity) as quantity_sold from orderitem group by productid order by sum(quantity) desc limit 10;
        select supplierid as competitors_id, companyname as competitor_name from product as p
        inner join supplier as s on p.supplierid = s.id
        where p.id in ( select productid from temp );
        
        
-- 3.	Create a combined list to display customers and suppliers details considering the following criteria 
-- 		a. Both customer and supplier belong to the same country
		select s.companyname as supplier, concat_ws( c.firstname, c.lastname) as customer, c.country as country
        from supplier as s
        inner join customer as c on s.country = c.country;
        
        
-- 		b. Customer who does not have supplier in their country
select concat_ws(firstname,lastname) as customer, country from customer where country not in 
( select distinct country from supplier );

-- 		c. Supplier who does not have customer in their country
select companyname as supplier, country from supplier where country not in 
( select distinct country from customer );


-- 4.	Every supplier supplies specific products to the customers.
-- Create a view of suppliers and total sales made by their products and write a query on this view to find out top 2 suppliers
-- (using windows function) in each country by total sales done by the products.
CREATE VIEW Supplier_Sales AS (
SELECT Supplier.Id SupplierId, CompanyName, Country, SUM(Quantity * OrderItem.UnitPrice) TotalSales
FROM Supplier
LEFT JOIN Product ON Product.SupplierId=Supplier.Id
LEFT JOIN OrderItem ON OrderItem.ProductId=Product.Id
GROUP BY 1,2,3
);

SELECT Country, CompanyName, Rnk TotalSalesRank FROM (
SELECT CompanyName, Country, RANK() OVER(PARTITION BY Country ORDER BY TotalSales DESC) Rnk
FROM Supplier_Sales) Base WHERE Rnk<=2 ORDER BY Country,Rnk;

-- 5.	Find out for which products, UK is dependent on other countries for the supply. List the countries which are supplying these products in the same list.
SELECT DISTINCT Product.Id, ProductName, Supplier.Country
FROM Product
JOIN OrderItem ON OrderItem.ProductId=Product.Id
JOIN Orders ON Orders.Id=OrderItem.OrderId
JOIN Customer ON Customer.Id=Orders.CustomerId 
JOIN  Supplier ON Supplier.Id=Product.SupplierId
WHERE Customer.Country='UK' AND Supplier.Country<>'UK';