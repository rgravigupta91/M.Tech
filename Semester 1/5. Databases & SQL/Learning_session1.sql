-- vidya@greatlearning.in

show databases;
create database session1;
use session1;

create table students
( rollno int,
stud_name varchar(15),
score float,
dept varchar(10));

desc students;

alter table students add column dob date;

alter table students add column age tinyint after score;	-- add a column after a particular column

alter table students add column id tinyint first;   		-- to add a column at first position 

alter table students drop column id;

alter table students modify column score int;

alter table students rename column dept to department;

rename table students to student;


alter table students change column score marks float;

alter table student modify column marks float after department;
drop table tablename;

drop database db_name;


------------ Truncate -----------------------
truncate table student;



----- Data Manipulation Language ---------------
insert into student values
(100,'Vidya', 40,'CS',58,'1970-01-01');


insert into student
(rollno, stud_name, age, department,dob)
values
(101,'A',30,'CS','2002-03-11');


update student set marks=40 where marks is NULL;

select count(*) from employees where department_id in ( 80, 90 );

select count(*) from employees where department_id <> 100;

select count(*) from employees where salary > 10000;

select count(*) from employees where manager_id is null;

select first_name, job_id from employees where first_name='neena' or first_name = 'bruce';

select first_name, job_id from employees where binary first_name in ('neena', 'bruce');

use hr;
select count(*) from employees
where salary between 10000 and 20000;

select first_name from employees where first_name like'_s%'

select *, salary * 12 as annual_salary from employees
where salary*12 > 100000


select 1+1 as result
select 1+1 as result from dual;

select distinct department_id from employees where department_id is not null

select distinct department_id, job_id from employees where department_id is not null

group by department_id




select department_id, first_name, hire_date
from employees
order by department_id, hire_date;




-- Display first name and last name from table 
-- whose last name contains 'LI' and arrange
-- first name in alphabetical order


select first_name, last_name from employees
where last_name like '%LI%'
order by first_name; 



select salary from employees order by salary desc;
select salary from employees order by salary desc limit 2;
select salary from employees order by salary desc limit 1,2;


help 'functions'
help 'round'

-- Functions -------------
round() 	: Rounds of the data
truncate()	: Truncates the data
floor()		: highest whole number less than the given number
ceil()		: lowest whole number greater than the given number
mod(10,3)	:
pow()		:
sqrt()		:
upper()		:
lower()		:

concat(str1,str2)
concat_ws('<shade>',str1,str2)	: seperated by shade
substring('welcome',2,3):	elc
length()				: actual storage bytes. Some characters like, latin, greek or chinese or some special characters, it takes more than 1 byte.
char_length()			: only number of characters visible in the attribute
instr('Hello','l')		: In string. Searches second string in first string and returns the first position of the str2 if it exists
lpad(salary, 10, *)		:
rpad(salary, 10, *)		:
replease('Jack' and 'Jue','J','BL')
trim()

trim('H' from 'Hello World')
trim(Trailing 'M' from 'MADAM')
trim(Leading 'M' from 'MADAM')
trim(Both 'M' from 'MADAM')
trim('M' from 'MADAM')			: equivalant to trim(Both 'M' from 'MADAM')
trim(Trailing 'M' from 'MADAM')


select first_name from emloyees
where binary first_name='Neena';		// Binary makes it case sensitive




select '1'+1;
select 1+'a';
select mod(10,3)



select 
	employee_id, 
    concat(first_name,' ', last_name) as Name, 
    job_id, length(last_name) as len, 
	instr(last_name,'a') as pos 
from employees
where job_id like '%REP%' and
last_name like '%a%'


select current_date();
select current_time();
select now();
select utc_timestamp();
select time_to_sec(now()), time_to_sec(current_time()), adddate(current_date() ,interval 50 day);
select time_to_sec(now()), time_to_sec(current_time()), adddate(current_date() ,interval 2 month);
select time_to_sec(now()), time_to_sec(current_time()), adddate(current_date() ,interval 4 year);

select date_sub(current_date(), interval 44 day);
select datediff(current_date(),'2024-01-01');
select datediff('2024-01-01',current_date());


select * from employees where adddate(hire_date, interval 25 year) <= current_date()

select year(current_date());
select year(now());

select date_format(current_date(),'%d-%m-%Y')
select date_format(current_date(),'%d-%m-%y')
select date_format(current_date(),'%d-%M-%Y')
select date_format(current_date(),'%c')

select date_format(current_date(),'%D %M %Y')

select str_to_date('13th April 2024','%D %M %Y')





-- April 14, 2024 -----------------
select cast('2018-10-31' as DATE);
select CONVERT('2018-10-31', DATE);


-- Control Flow Statements ------------
-- If

select if(mod(10,2)=0,'Even Number', 'Odd Number');

-- ifnull
use hr;
select commission_pct, ifnull(commission_pct,.01) from employees;
select commission_pct, coalesce(commission_pct,.01) from employees;


select first_name, salary, commission_pct,
salary*commission_pct as comm_value,
salary+(salary*commission_pct) as total_salary
from employees;


select first_name, salary, commission_pct,
salary*commission_pct as comm_value,
salary+(salary*ifnull(commission_pct,0)) as total_salary
from employees;


-- nullif --
-- nullif( exp1, exp2 ) if exp1 = exp2 returns null otherwise returns exp1
select job_id, nullif(job_id,'IT_PROG') from employees;


select first_name, salary,
	case 
		when salary > 20000 then 'A+'
        when salary > 18000 then 'A'
        when salary > 15000 then 'B+'
        when salary > 12000 then 'B'
        when salary > 10000 then 'C'
        else 'Need Revision' end as salary_grade from employees;
        
        
-- Constraints --
create table author
( id int primary key,
name varchar(10) not null default 'Mr.X',
country varchar(10) not null check (country in ('USA', 'UK', 'UAE')),
page int check(page>0),
p_no varchar(10) not null unique
);

describe table author;
show create table author;
-- CREATE TABLE `author` (
--    `id` int NOT NULL,
--    `name` varchar(10) NOT NULL DEFAULT 'Mr.X',
--    `country` varchar(10) NOT NULL,
--    `page` int DEFAULT NULL,
--    `p_no` varchar(10) NOT NULL,
--    PRIMARY KEY (`id`),
--    UNIQUE KEY `p_no` (`p_no`),
--    CONSTRAINT `author_chk_1` CHECK ((`country` in (_utf8mb4'USA',_utf8mb4'UK',_utf8mb4'UAE'))),
--    CONSTRAINT `author_chk_2` CHECK ((`page` > 0))
--  ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
--  
insert into author values(1,'XYZ','USA',-20,1);


select '*' as result1
union
select '**' as result2
union
select '***' as result3;
	
select department_id, min(salary) from employees
group by department_id
order by min(salary);

select min(salary) from employees where department_id = 10;
select department_id, max(count(*)) from employees group by department_id;
select department_id, count(*) from employees where department_id is not null group by department_id;
select department_id, count(commission_pct) cnt from employees where department_id is not null group by department_id having cnt > 0;
select count(commission_pct) from employees;

select * from employees;
select * from employees where department_id is null;

-- Execution Sequence
-- From 
-- Where
-- Group By
-- having
-- Select
-- Order By
-- Limit


select distinct year(hire_date), count(year(hire_date)) from employees group by year(hire_date) order by count(year(hire_date)) desc limit 1;

USE HR;
SELECT DEPARTMENT_ID, COUNT(*) AS NUM FROM EMPLOYEES WHERE SALARY > 10000 AND YEAR(HIRE_DATE) = 1997 group by DEPARTMENT_ID;

#Joins
-- Inner Join
-- Left Outer Join
-- Right Outer Join
-- Full Outer Join

-- Self join
-- cross join
-- natural joins

select distinct(department_id) from employees where department_id is not null;


select distinct(d.department_name) from employees as e
right outer join departments as d on e.department_id = d.department_id
where e.department_id is null;



select month(e.hire_date), count(*) from employees as e
left outer join departments as d on e.department_id = d.department_id
left outer join locations as l   on l.location_id = d.location_id
where l.city = 'seattle'
group by month(e.hire_date)
having count(*)>2;


use hr;
-- This query will pick the second row from the selection list.
select salary from employees order by salary desc limit 1,1;


-- earning more than any one of the david
select * from employees where salary > ANY  -- or
(select salary from employees where first_name='David')


-- earning more than all of the davids.
select * from employees where salary > ALL  -- AND
(select salary from employees where first_name='David')


select * from employees e where salary >
(select avg(salary)a from employees where department_id = e.department_id);

select * from employees where exists
(select * from employees where department_id=80);

-- List all department name where employees are working
select department_name from departments as d where exists
(select distinct department_id from employees where department_id = d.department_id);


select department_id, case department_id when 80 then true else false end as exist from employees;



-- Display the name & department ID of all departments that has atleast one employee with salary greater than 10,000
-- ( Use both in and exists operators )
select department_id, department_name from departments as d where exists
(select true from employees where salary > 10000 and department_id = d.department_id limit 1)



-- Find the highest average salary among all the department
-- without using limit


select max(a.avg_salary) as highest_avg_salary from 
(select department_id, avg(salary) as avg_salary from employees group by department_id)
 as a;
 
 -- Other way
 select department_id, avg(salary) avg_sal from employees
 group by department_id
 having avg_sal>=ALL
 ( select avg(salary) from employees
 group by department_id);
 
 
 
 
 
 -- ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 -- Select Clause subquery
 -- ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 
 select count(employee_id) as emp_count, (select count(department_id) from departments ) as dept_count
 from employees;
 
  select employee_id, (select department_name from departments where department_id=e.department_id ) as dept_count
 from employees as e;
 
 
 
 -- display employee name and manager
 select e.first_name as emp_name, ( select m.first_name from employees as m where m.employee_id = e.manager_id ) as mngr_name from employees as e;
 
 
 
 -- Subquery in from clause/inline query/derived table
 select * from
 (select first_name, salary, salary*0.1 bonus from employees) temp
 where bonus > 1000;
 
 
 -- CTE with clause
 with temp as
 ( select first_name, salary, salary*0.1 bonus from employees)
 select * from temp where bonus>1000;
 
 
/* 
 Sub Queries
 	Types:
		Single Value / Row
		Multiple Row
		Multiple Columns
		corelated S.Q.  -> Sub Query depending on the outer query
		Multilevel / Nested S.Q.
			32 / No Limit
 	Operators:
 		=
        IN
        NOT IN
        ANY
        ALL
        
        
ETL - Extract Transform Load
		Sources
			Structured 			-	Table 
            Semi Structured		-	XML, JSON
            Unstructured 		-	Flat, 



Advanced Aggregate Function
	Window Functions
		Over()
			Order by
            partition by / Group by
            Framing / Slicing
		
        Ranking					Aggregate				Analytical
			Row_number()
            Rank()
            Dense_rank()
            c
		Aggregate
        Analytical
			Lead()
            Lag()
            F_value()
            L_Value()
            N_
*/


use hr;

-- row_number()

-- List only the records between 20 and 30
select * from
( select row_number()over() as 'S.No', first_name, salary
from employees) temp
where `S.No` between 20 and 30;

-- or

select * from
( select row_number()over() as 'S.No', first_name, salary
from employees) temp
limit 19,11;


-- Rank()
select first_name, salary, rank()over(order by salary desc) as Rnk from employees;

select first_name, salary, dense_rank()over(order by salary desc) as Rnk from employees;

select first_name, salary, cume_dist()over(order by salary asc) as cum_dist from employees; 





select first_name, salary,
rank()over(order by salary desc) as Rnk1,
dense_rank()over(order by salary desc) as Rnk2,
cume_dist()over(order by salary desc) as cum_dist
from employees;


-- aggregate Functions using Over()
select first_name, salary,
max(salary)over() as max_sal
from employees;


-- List employees earning more than the average salary from 
select * from
(select first_name, salary, avg(salary)over() as avg_sal
from employees) temp
where salary > avg_sal;

select * from
( select *,
avg(salary)over(partition by department_id) as avg_salary
from employees) t
where salary > avg_salary;



-- list all seniors from each department
select * from
( select *, min(hire_date)over(partition by department_id) as h_date from employees ) as t
where hire_date = h_date;




-- Lag() and Lead()

select first_name, salary,
lead(salary)over() as next_sal
from employees;

select first_name, salary,
lag(salary)over() as prev_sal
from employees;

select first_name, salary,
lag(salary,2)over() as prev2_sal
from employees;


-- employee_id, first_job, promoted_job
select * from					-- This will lead to wrong output. Check why?
( select employee_id, start_date, 
job_id as first_job, 
lead(job_id)over(partition by employee_id) as promoted_job 
from job_history order by employee_id, start_date) as t
where promoted_job is not null;

-- or
select * from
(select employee_id, start_date, job_id as first_job,
lead(job_id)over(partition by employee_id order by start_date asc) as promoted_job from job_history) as t
where promoted_job is not null;



-- Find the appointment frequency for each department
select department_id,
avg(diff_days) as freq from
(select *,
coalesce(datediff(lead(hire_date)over(partition by department_id), hire_date),0) as diff_days from employees) as temp
-- where diff_days is not null
group by department_id;




-- Framing and slicing

-- F_VALUE() L_VALUE()
select first_name, salary,
last_value(salary)over(order by salary desc) l_val
from employees;

select first_name, salary,
last_value(salary)over(order by salary desc
rows between unbounded preceding and unbounded following) l_val
from employees;


-- rows between unbounded preceding and unbounded following
-- rows between unbounded preceding and current row
-- rows between current row and unbounded following
-- rows between 1 preceding and 1 following



-- running total of salary
select first_name, salary, sum(salary)over(rows between unbounded preceding and current row) as running_total from employees;

help `sum`;




-- Exercise
-- Val	Res
-- 10	NULL
-- 11	NULL
-- 12	NULL
-- 13	33
-- 14	36
-- 15	39
-- 16	42

-- create t able t1(val int);
-- insert into t1 values(10);
-- insert into t1 values(11);
-- insert into t1 values(12);
-- insert into t1 values(13);
-- insert into t1 values(14);
-- insert into t1 values(15);
-- insert into t1 values(16);

select val, if(val<13,NULL,res) as res from
( select val, lag(val,2)over()*3 as res from t1) temp;


set @no_of_row = 3;

select val, 
	case 
		when row1 > @no_of_row 
			then sum(val)over(rows between norow preceding and 1 preceding ) 
		else 
			NULL end res from
( select row_number()over() row1, val, @no_of_row as norow from t1 ) t;




-- ntile()
select bucket, count(*) from
( select first_name, ntile(10)over() bucket from employees) t
group by bucket;
