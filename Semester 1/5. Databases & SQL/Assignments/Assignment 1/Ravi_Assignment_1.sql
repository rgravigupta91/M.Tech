# Pre-Requisites:
#Step 1:
create schema ASSIGNMENT1;
USE ASSIGNMENT1;
Create table new_cricket
( Player_Id	CHAR(5),
Player_Name	CHAR(20),
Runs	INT,
Popularity	INT,
Charisma INT
) ;

#Step 2:
Insert into NEW_CRICKET values ( 'PL1', 'Virat', 50, 10, 55);
Insert into NEW_CRICKET values ( 'PL2', 'Rohit', 41, 7, 30);
Insert into NEW_CRICKET values ( 'PL3', 'Jadeja', 33, 6, 24);
Insert into NEW_CRICKET values ( 'PL4', 'Dhoni', 35, 15, 59);
Insert into NEW_CRICKET values ( 'PL6', 'Yadav', 66, 10, 35);
Insert into NEW_CRICKET values ( 'PL8', 'Binny', 44, 11, 32);
Insert into NEW_CRICKET values ( 'PL9', 'Rayudu', 63, 12, 39);
Insert into NEW_CRICKET values ( 'PL5', 'Dhawan', 45, 12, NULL);
Insert into NEW_CRICKET values ( 'PL7', 'Raina', 32, 9, NULL);
Insert into NEW_CRICKET values ( 'PL10', 'rahane', 21, 10, NULL);
Insert into NEW_CRICKET values ( 'PL11', 'A. Patel', 12, 9, NULL);
Insert into NEW_CRICKET values ( 'PL12', 'B. Kumar', 30, 7, NULL);

# Question 1:

-- --------------------------------------------------------
# Dataset Used: new_cricket.csv
-- --------------------------------------------------------

# Q1. Extract the Player_Id and Player_name of the players where the charisma value is null.
SELECT Player_Id, Player_Name from new_cricket where Charisma is NULL;
 
# Q2. Write a MySQL query to display all the NULL values in the column Charisma imputed with 0.
select Player_Id, Player_Name, Runs, Popularity, ifnull(Charisma, 0) as Charisma from new_cricket;

-- --------------------------------------------------------
# Dataset Used: churn1.csv 
-- -------------------------------------------------------
# select the csv file and create table using Manu -> Database-> Reverse Engineer Database

# Q3. Rename the table churn1 to “Churn_Details”.
alter table churn1 rename churn_details;

# Q4. Write a query to create a new column new_Amount that contains the sum of TotalAmount and MonthlyServiceCharges.
alter table churn_details add new_Amount int as ( TotalAmount + MonthlyServiceCharges );


# Q5. Rename column new_Amount to Amount.
alter table churn_details rename column new_Amount to Amount;

# Q6. Drop the column “Amount” from the table “Churn_Details”.
alter table churn_details drop column Amount;

# Q7. Write a query to extract the customerID, InternetConnection and gender 
# from the table “Churn_Details ” where the value of the column “InternetConnection” has ‘i’ 
# at the second position.
select customerID, InternetConnection, gender 
from Churn_Details
where InternetConnection like '_i%';


# Q8. Find the records where the tenure is 6x, where x is any number.(107 Rows)
select * from churn_details where tenure like '6_';

#Q9. Write a query to display the player names in capital letter and arrange in alphabatical order along with the charisma, display 0 for whom the charisma value is NULL.
select upper(Player_Name) as Player_Name, ifnull(Charisma,0) as Charisma from new_Cricket order by Player_Name;


# Pre-Requisites:
# Step 1 : Create table as below.

Create table Bank_Inventory_pricing 
( Product CHAR(15) , 
Quantity INT, 
Price  Real ,
purchase_cost Decimal(6,2), 
Estimated_sale_price  Float,
Month int) ;

# Step2: 
# Insert records for above 
 Insert into Bank_Inventory_pricing values ( 'PayCard'   , 2 , 300.45, 8000.87, 9000.56, 1 ) ;
 Insert into Bank_Inventory_pricing values ( 'PayCard'   , 2 , 800.45, 5000.80, 8700.56, 1 ) ;
 Insert into Bank_Inventory_pricing values ( 'PayCard'   , 2 , 500.45, 6000.47, 7400.56, 1 ) ;
 Insert into Bank_Inventory_pricing values ( 'PayPoints' , 4 , 390.87, 7000.67, 6700.56, 2)  ;
 Insert into Bank_Inventory_pricing values ( 'SmartPay' , 5  , 290.69, 5600.77, 3200.12 , 1)  ;
 Insert into Bank_Inventory_pricing values ( 'MaxGain',    3 , NULL, 4600.67, 3233.11 , 1 ) ;
 Insert into Bank_Inventory_pricing values ( 'MaxGain',    6 , 220.39, 4690.67, NULL , 2 ) ;
 Insert into Bank_Inventory_pricing values ( 'SuperSave', 7 , 290.30, NULL, 3200.13 ,1 ) ;
 Insert into Bank_Inventory_pricing values ( 'SuperSave', 6 , 560.30, NULL, 4200.13 ,1 ) ;
 Insert into Bank_Inventory_pricing values ( 'SuperSave', 6 , NULL, 2600.77, 3200.13 ,2 ) ;
 Insert into Bank_Inventory_pricing values ( 'SuperSave', 9 , NULL, 5400.71, 9200.13 ,3 ) ;
 Insert into Bank_Inventory_pricing values ( 'SmartSav',   3 , 250.89, 5900.97, NULL ,1 ) ;
 Insert into Bank_Inventory_pricing values ( 'SmartSav',   3 , 250.89, 5900.97, 8999.34 ,1 ) ;
 Insert into Bank_Inventory_pricing values ( 'SmartSav',   3 , 250.89, NULL , 5610.82 , 2 ) ;
 Insert into Bank_Inventory_pricing values ( 'EasyCash',   3 , 250.89, NULL, 5610.82 ,1 ) ;
 Insert into Bank_Inventory_pricing values ( 'EasyCash',   3 , 250.89, NULL, 5610.82 , 2 ) ;
 Insert into Bank_Inventory_pricing values ( 'EasyCash',   3 , 250.89, NULL, 5610.82 , 3 ) ;
 Insert into Bank_Inventory_pricing values ( "BusiCard"  ,  1, 3000.99 , NULL, 3500, 3) ; 
 Insert into Bank_Inventory_pricing values ( "BusiCard"  ,  1, 4000.99 , NULL, 3500, 2) ; 

# Create table
Create table Bank_branch_PL
(Branch   varchar(15),
  Banker   Int,
  Product varchar(15) ,
  Cost  Int,
  revenue Int,
  Estimated_profit Int,   
  month  Int);
Insert into Bank_branch_PL values ( 'Delhi', 99101, 'SuperSave', 30060070, 50060070,  20050070, 1 ) ;
Insert into Bank_branch_PL values ( 'Delhi', 99101, 'SmartSav',   45060070, 57060070, 30050070, 2) ;
Insert into Bank_branch_PL values ( 'Delhi', 99101, 'EasyCash',   66660070, 50090090, 10050077, 3) ;
Insert into Bank_branch_PL values ( 'Hyd', 99101, 'SmartSav',   66660070, 79090090, 10050077, 3) ;
Insert into Bank_branch_PL values ( 'Banglr', 77301, 'EasyCash',   55560070, 61090090, 9950077, 3) ;
Insert into Bank_branch_PL values ( 'Banglr', 77301, 'SmartSav',   54460090, 53090080, 19950077, 3) ;
Insert into Bank_branch_PL values ( 'Hyd', 77301, 'SmartSav',   53060090, 63090080, 29950077, 3) ;
Insert into Bank_branch_PL values ( 'Hyd', 88201, 'BusiCard',  	40030070, 60070080, 10050070,1) ;
Insert into Bank_branch_PL values ( 'Hyd', 88201, 'BusiCard',  	70030070, 60070080, 25060070,1) ;
Insert into Bank_branch_PL values ( 'Hyd', 88201, 'SmartSav', 	40054070, 60070080, 20050070, 2) ;
Insert into Bank_branch_PL values ( 'Banglr', 99101, 'SmartSav',   88660070, 79090090, 10050077, 3) ;


############################################ ############################################



# 10) Print average of estimated_sale_price upto two decimals from bank_inventory_pricing table.
/* Solution */
select round(avg(Estimated_sale_price),2) as avg_est_sp from bank_inventory_pricing;

# Question 11:
# 11. Print Products that are appearing more than once in bank_inventory_pricing and whose purchase_cost is
# greater than  estimated_sale_price , assuming estimated_sale_price is 0 when there is no value given
/* Solution */
select Product, count(*) as count 
from bank_inventory_pricing 
where purchase_cost > ifnull(Estimated_sale_price,0) 
group by Product 
having count(*) > 1;


# 12) Print unique records of bank_inventory_pricing without displaying the month.
/* Solution */
select distinct Product, Quantity, Price, purchase_cost, Estimated_sale_price
from bank_inventory_pricing;

# 13) Print the count of unique Products used in  Bank_inventory_pricing
/* Solution */
select count(distinct Product) from bank_inventory_pricing;

# 14) Print the sum of Purchase_cost of Bank_inventory_pricing during 1st and 2nd month
/* Solution */
select sum(purchase_cost) as total_purchase_cost from bank_inventory_pricing where month in ( 1, 2 );

# 15) Print  products whose average of purchase_cost is less than sum of purchase_cost of  Bank_inventory_pricing
/* Solution */
select Product, avg(purchase_cost) as avg_cost, sum(purchase_cost) as total_cost from bank_inventory_pricing 
group by Product
having avg(purchase_cost) < sum(purchase_cost);


