create schema assignment3;
use assignment3;
-----------------------------------------------------------------
# Dataset Used: students.csv
-- --------------------------------------------------------------

# Q1. Display the records from the students table where partition should be done 
-- on subjects and use sum as a window function on marks, 
-- -- arrange the rows in unbounded preceding manner.
select distinct subject, sum(marks)over(partition by subject) from students;


# Q2. Find the dense rank of the students on the basis of their marks subjectwise. Store the result in a new table 'Students_Ranked'
select student_id, subject, name, marks, dense_rank()over(partition by subject order by marks desc) from students;

-----------------------------------------------------------------
# Dataset Used: website_stats.csv and web.csv
-----------------------------------------------------------------
# Q3. Display the difference in ad_clicks between the current day and the next day for the website 'Olympus'
select website_id, day, (ad_clicks - lead(ad_clicks)over(partition by website_id)) as diff_ad_click from website_stats where website_id = 
(select id from web where name = 'Olympus') ; 

# Q4. Write a query that displays name of the website and it's launch date. The query should also display the date of recently launched website in the third column.
select name, launch_date, lag(launch_date)over(order by launch_date asc) as recent_launch_website_date from web;

-----------------------------------------------------------------
# Dataset Used: play_store.csv and sale.csv
-----------------------------------------------------------------
# Q5. Write a query thats orders games in the play store into three buckets as per editor ratings received  
select *, case
	when editor_rating > 8 then 'Excellent'
    when editor_rating > 5 then 'Good'
    else 'Average' end as Rating from play_store where id in ( select distinct game_id from sale );


# Q6. Write a query that displays the name of the game, the price, the sale date and the 4th column should display 
# the sales consecutive number i.e. ranking of game as per the sale took place, so that the latest game sold gets number 1. Order the result by editor's rating of the game

select t.game_id, p.name, s.price, t.latest_sale_date as sale_date, rank()over(order by latest_sale_date desc) as sales_consecutive_number from
(select distinct game_id, max(date) as latest_sale_date from sale group by game_id) as t
left outer join sale as s on t.game_id = s.game_id and t.latest_sale_date = s.date
left outer join play_store as p on t.game_id = p.id
order by p.editor_rating desc;

-----------------------------------------------------------------
# Dataset Used: movies.csv, customers.csv, ratings.csv, rent.csv
-----------------------------------------------------------------
# Q7. For each movie, show the following information: title, genre, average user rating for that movie 
# and its rank in the respective genre based on that average rating in descending order (so that the best movies will be shown first).

select *, dense_rank()over(partition by genre order by avg_rating desc) from
(select title, genre, coalesce(avg(r.rating), 0) as avg_rating from movies as m
left outer join ratings as r on m.id = r.movie_id
group by title, genre) as temp;


# Q8. For each rental date, show the rental_date, the sum of payment amounts (column name payment_amounts) from rent 
#on that day, the sum of payment_amounts on the previous day and the difference between these two values.

select 
	rental_date, 
    payment_amounts, 
    coalesce( lag(payment_amounts)over(), 0 ) as prev_day_tot_payment ,
    payment_amounts - coalesce( lag(payment_amounts)over(), 0 ) as diff_amount
from
( select 
	rental_date, 
    sum(payment_amount) as payment_amounts
from rent 
group by rental_date) as temp;




-- ---------------------------------------------------------------------------------------------------------------------------------------------------
-- Q9. Create a table Passengers. Follow the instructions given below: 
-- -- Q9.1. Values in the columns Traveller_ID and PNR should not be null.
-- -- Q9.2. Only passengers having age greater than 18 are allowed.
-- -- Q9.3. Assign primary key to Traveller_ID
-- ---------------------------------------------------------------------------------------------------------------------------------------------------

-- 9.
create table Passengers(
	Traveller_ID char(2) primary key,
    Name   char(30),
    age smallint check( age > 18 ),
    PNR char(10) not null,
    Flight_ID smallint,
    Ticket_price int
);

-- -----------------------------------------------------
-- Table Passengers
-- -----------------------------------------------------


-- ---------------------------------------------------------------------------------------------------------------------------------------------------
-- Questions for table Passengers:  
-- -- Q9.4. PNR status should be unique and should not be null.
-- -- Q9.5. Flight Id should not be null.
-- ---------------------------------------------------------------------------------------------------------------------------------------------------
alter table Passengers modify column PNR char(10) not null unique;
alter table Passengers modify column Flight_ID smallint not null;


-- ---------------------------------------------------------------------------------------------------------------------------------------------------
-- Q10. Create a table Senior_Citizen_Details. Follow the instructions given below: 
-- -- Q10.1. Column Traveller_ID should not contain any null value.
-- -- Q10.2. Assign primary key to Traveller_ID
-- -- Q10.3. Assign foreign key constraint on Traveller_ID such that if any row from passenger table is updated, then the Senior_Citizen_Details table should also get updated.
-- -- --  Also deletion of any row from passenger table should not be allowed.
-- ---------------------------------------------------------------------------------------------------------------------------------------------------

-- -----------------------------------------------------
-- Table Senior_Citizen_Details
-- -----------------------------------------------------
create table Senior_Citizen_Details(
	Traveller_ID char(2) primary key,
    Senior_Citizen char(2),
    Discounted_Price smallint,
    CONSTRAINT `scd_1` FOREIGN KEY (`Traveller_ID`) REFERENCES `Passengers` (`Traveller_ID`) on update cascade
);
-- --------------------------------------------------------------------------------------------------------------------------------------------------- 

   
-- ---------------------------------------------------------------------------------------------------------------------------------------------------
-- Q11. Create a table books. Follow the instructions given below: 
-- -- Columns: books_no, description, author_name, cost
-- -- Q11.1. The cost should not be less than or equal to 0.
-- -- Q11.2. The cost column should not be null.
-- -- Q11.3. Assign a primary key to the column books_no.
-- ---------------------------------------------------------------------------------------------------------------------------------------------------

create table books(
	book_no char(10) primary key,
    description char(30),
    author_name char(30),
    cost smallint not null check( cost > 0 )
);



-- ----------------------------------------------------
# Datasets Used: employee_details.csv and department_details.csv
-- ----------------------------------------------------

 
 
# Q.12 Create a view "pattern_matching" such that it contains the records from the columns employee_id, first_name, job_id and salary from the table name "employee_details" 
-- --  where first_name ends with "l".
#Solution:

create view pattern_matching as 
(select employee_id, first_name, job_id, salary
from employee_details
where first_name like '%l');
 
 
# Q.13 Drop multiple existing views "pattern_matching".
#Solution:

 drop view pattern_matching;
 
# Q.14 Create a view "employee_department" that contains the common records from the tables "employee_details" and "department_table".
#Solution:
create view employee_department as 
(select e.employee_id,
e.first_name,
e.last_name,
e.email,
e.phone_number,
e.hire_date,
e.job_id,
e.salary,
e.manager_id,
e.department_id,
d.department_name,
d.location_id from employee_details as e
inner join department_details as d
on e.department_id = d.department_id and e.employee_id = d.employee_id);

-- ----------------------------------------------------
# Datset Used: admission_predict.csv
-- ----------------------------------------------------
# Q.15 A university focuses only on SOP and LOR score and considers these scores of the students who have a research paper. 
#Create a view for that university.
#Solution:
create view research_admission as
(select SOP, LOR, research, `chance of admit` from admission_predict);
 
