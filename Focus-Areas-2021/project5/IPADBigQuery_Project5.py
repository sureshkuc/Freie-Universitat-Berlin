from google.colab import auth
auth.authenticate_user()
print('Authenticated')

%load_ext google.colab.data_table

project_id = 'centered-oasis-297411'


%env GCLOUD_PROJECT=centered-oasis-297411

# Source: https://github.com/SohierDane/BigQuery_Helper/blob/master/bq_helper.py

import time
import pandas as pd

from google.cloud import bigquery


class BigQueryHelper(object):
    """
    Helper class to simplify common BigQuery tasks like executing queries,
    showing table schemas, etc without worrying about table or dataset pointers.
    See the BigQuery docs for details of the steps this class lets you skip:
    https://googlecloudplatform.github.io/google-cloud-python/latest/bigquery/reference.html
    """

    def __init__(self, active_project, dataset_name, max_wait_seconds=180):
        self.project_name = active_project
        self.dataset_name = dataset_name
        self.max_wait_seconds = max_wait_seconds
        self.client = bigquery.Client()
        self.__dataset_ref = self.client.dataset(self.dataset_name, project=self.project_name)
        self.dataset = None
        self.tables = dict()  # {table name (str): table object}
        self.__table_refs = dict()  # {table name (str): table reference}
        self.total_gb_used_net_cache = 0
        self.BYTES_PER_GB = 2**30

    def __fetch_dataset(self):
        """
        Lazy loading of dataset. For example,
        if the user only calls `self.query_to_pandas` then the
        dataset never has to be fetched.
        """
        if self.dataset is None:
            self.dataset = self.client.get_dataset(self.__dataset_ref)

    def __fetch_table(self, table_name):
        """
        Lazy loading of table
        """
        self.__fetch_dataset()
        if table_name not in self.__table_refs:
            self.__table_refs[table_name] = self.dataset.table(table_name)
        if table_name not in self.tables:
            self.tables[table_name] = self.client.get_table(self.__table_refs[table_name])

    def __handle_record_field(self, row, schema_details, top_level_name=''):
        """
        Unpack a single row, including any nested fields.
        """
        name = row['name']
        if top_level_name != '':
            name = top_level_name + '.' + name
        schema_details.append([{
            'name': name,
            'type': row['type'],
            'mode': row['mode'],
            'fields': pd.np.nan,
            'description': row['description']
                               }])
        # float check is to dodge row['fields'] == np.nan
        if type(row.get('fields', 0.0)) == float:
            return None
        for entry in row['fields']:
            self.__handle_record_field(entry, schema_details, name)

    def __unpack_all_schema_fields(self, schema):
        """
        Unrolls nested schemas. Returns dataframe with one row per field,
        and the field names in the format accepted by the API.
        Results will look similar to the website schema, such as:
            https://bigquery.cloud.google.com/table/bigquery-public-data:github_repos.commits?pli=1
        Args:
            schema: DataFrame derived from api repr of raw table.schema
        Returns:
            Dataframe of the unrolled schema.
        """
        schema_details = []
        schema.apply(lambda row:
            self.__handle_record_field(row, schema_details), axis=1)
        result = pd.concat([pd.DataFrame.from_dict(x) for x in schema_details])
        result.reset_index(drop=True, inplace=True)
        del result['fields']
        return result

    def table_schema(self, table_name):
        """
        Get the schema for a specific table from a dataset.
        Unrolls nested field names into the format that can be copied
        directly into queries. For example, for the `github.commits` table,
        the this will return `committer.name`.
        This is a very different return signature than BigQuery's table.schema.
        """
        self.__fetch_table(table_name)
        raw_schema = self.tables[table_name].schema
        schema = pd.DataFrame.from_dict([x.to_api_repr() for x in raw_schema])
        # the api_repr only has the fields column for tables with nested data
        if 'fields' in schema.columns:
            schema = self.__unpack_all_schema_fields(schema)
        # Set the column order
        schema = schema[['name', 'type', 'mode', 'description']]
        return schema

    def list_tables(self):
        """
        List the names of the tables in a dataset
        """
        self.__fetch_dataset()
        return([x.table_id for x in self.client.list_tables(self.dataset)])

    def estimate_query_size(self, query):
        """
        Estimate gigabytes scanned by query.
        Does not consider if there is a cached query table.
        See https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs#configuration.dryRun
        """
        my_job_config = bigquery.job.QueryJobConfig()
        my_job_config.dry_run = True
        my_job = self.client.query(query, job_config=my_job_config)
        return my_job.total_bytes_processed / self.BYTES_PER_GB

    def query_to_pandas(self, query):
        """
        Execute a SQL query & return a pandas dataframe
        """
        my_job = self.client.query(query)
        start_time = time.time()
        while not my_job.done():
            if (time.time() - start_time) > self.max_wait_seconds:
                print("Max wait time elapsed, query cancelled.")
                self.client.cancel_job(my_job.job_id)
                return None
            time.sleep(0.1)
        # Queries that hit errors will return an exception type.
        # Those exceptions don't get raised until we call my_job.to_dataframe()
        # In that case, my_job.total_bytes_billed can be called but is None
        if my_job.total_bytes_billed:
            self.total_gb_used_net_cache += my_job.total_bytes_billed / self.BYTES_PER_GB
        return my_job.to_dataframe()

    def query_to_pandas_safe(self, query, max_gb_scanned=1):
        """
        Execute a query, but only if the query would scan less than `max_gb_scanned` of data.
        """
        query_size = self.estimate_query_size(query)
        if query_size <= max_gb_scanned:
            return self.query_to_pandas(query)
        msg = "Query cancelled; estimated size of {0} exceeds limit of {1} GB"
        print(msg.format(query_size, max_gb_scanned))

    def head(self, table_name, num_rows=5, start_index=None, selected_columns=None):
        """
        Get the first n rows of a table as a DataFrame.
        Does not perform a full table scan; should use a trivial amount of data as long as n is small.
        """
        self.__fetch_table(table_name)
        active_table = self.tables[table_name]
        schema_subset = None
        if selected_columns:
            schema_subset = [col for col in active_table.schema if col.name in selected_columns]
        results = self.client.list_rows(active_table, selected_fields=schema_subset,
            max_results=num_rows, start_index=start_index)
        results = [x for x in results]
        return pd.DataFrame(
data=[list(x.values()) for x in results], columns=list(results[0].keys()))

bq_assistant = BigQueryHelper("bigquery-public-data", "github_repos")
bq_assistant.list_tables()
bq_assistant.table_schema("licenses")
bq_assistant.head("licenses", num_rows=10)

QUERY1 = """
SELECT
  call.name AS call_name,
  COUNT(call.name) AS call_count_for_call_set
FROM
  `bigquery-public-data.human_genome_variants.platinum_genomes_deepvariant_variants_20180823` v, v.call
GROUP BY
  call_name
ORDER BY
  call_name
"""

bq_assistant.estimate_query_size(QUERY1)  

df1 = bq_assistant.query_to_pandas_safe(QUERY1,max_gb_scanned = 2)

df1

QUERY2="""
SELECT
  call.name AS call_name,
  COUNT(call.name) AS call_count_for_call_set
FROM
  `bigquery-public-data.human_genome_variants.platinum_genomes_deepvariant_variants_20180823` v, v.call
WHERE
  EXISTS (SELECT 1 FROM UNNEST(call.genotype) AS gt WHERE gt > 0)
  AND NOT EXISTS (SELECT 1 FROM UNNEST(call.genotype) AS gt WHERE gt < 0)
GROUP BY
  call_name
ORDER BY
  call_name
  """



bq_assistant.estimate_query_size(QUERY2) 

df2 = bq_assistant.query_to_pandas_safe(QUERY2,max_gb_scanned = 5)
df2

QUERY3 = """
SELECT
  call_filter,
  COUNT(call_filter) AS number_of_calls
FROM
  `bigquery-public-data.human_genome_variants.platinum_genomes_deepvariant_variants_20180823` v,
  v.call,
  UNNEST(call.FILTER) AS call_filter
GROUP BY
  call_filter
ORDER BY
  number_of_calls
  """

bq_assistant.estimate_query_size(QUERY3)    

df3 = bq_assistant.query_to_pandas_safe(QUERY3)

df3.head()

QUERY4= """       
  SELECT
    reference_name,
    COUNT(reference_name) AS number_of_variant_rows
  FROM
    `bigquery-public-data.human_genome_variants.platinum_genomes_deepvariant_variants_20180823` v
  WHERE
    EXISTS (SELECT 1
            FROM UNNEST(v.call) AS call, UNNEST(call.genotype) AS gt
          WHERE gt > 0)
  GROUP BY
    reference_name
  ORDER BY
    CASE
      WHEN SAFE_CAST(REGEXP_REPLACE(reference_name, '^chr', '') AS INT64) < 10
        THEN CONCAT('0', REGEXP_REPLACE(reference_name, '^chr', ''))
        ELSE REGEXP_REPLACE(reference_name, '^chr', '')
    END
       """

bq_assistant.estimate_query_size(QUERY4)    

df4 = bq_assistant.query_to_pandas_safe(QUERY4,max_gb_scanned = 5)


df4.head()

QUERY5="""
SELECT
  REGEXP_REPLACE(reference_name, '^chr', '') AS chromosome,
  COUNT(reference_name) AS number_of_variant_rows
FROM
  `bigquery-public-data.human_genome_variants.platinum_genomes_deepvariant_variants_20180823` v
WHERE
  EXISTS (SELECT 1
            FROM UNNEST(v.call) AS call, UNNEST(call.genotype) AS gt
          WHERE gt > 0)
GROUP BY
  chromosome
ORDER BY
  chromosome
  """

bq_assistant.estimate_query_size(QUERY5) 

df5 = bq_assistant.query_to_pandas_safe(QUERY5,max_gb_scanned = 5)
df5

# Count the number of samples in the phenotypic data
QUERY1="""
SELECT
  COUNT(sample) AS all_samples,
  SUM(IF(In_Phase1_Integrated_Variant_Set = TRUE, 1, 0)) AS samples_in_variants_table
FROM
  `genomics-public-data.1000_genomes.sample_info`
  """

bq_assistant.estimate_query_size(QUERY1) 

df = client.query(QUERY1).to_dataframe()

df.head()

import matplotlib.pyplot as plt

#From DataFrame
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
samples = list(df.columns)
values = df.iloc[0,:].values
ax.bar(samples, values, color = ['red', 'cyan'])
plt.show()

QUERY1="""
SELECT
  gender,
  gender_count,
  (gender_count/SUM(total_count) OVER (PARTITION BY gender)) as gender_ratio
FROM (
  SELECT
    gender,
    COUNT(gender) AS gender_count
  FROM
    `genomics-public-data.1000_genomes.sample_info`
  WHERE
    In_Phase1_Integrated_Variant_Set = TRUE
  GROUP BY gender
    ),(select count(gender) as total_count from  `genomics-public-data.1000_genomes.sample_info` where In_Phase1_Integrated_Variant_Set = TRUE)
"""

bq_assistant.estimate_query_size(QUERY1) 

df1 = bq_assistant.query_to_pandas_safe(QUERY1,max_gb_scanned = 3)
df1

import matplotlib.pyplot as plt


labels = df1["gender"]
sizes = df1["gender_count"]


fig1, ax1 = plt.subplots()
ax1.pie(sizes,  labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=0)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

QUERY2= """
SELECT
  population,
  population_description,
  population_count,
  (population_count/SUM(total_count) OVER (PARTITION BY population))  AS population_ratio,
  super_population,
  super_population_description,
from(
  SELECT
    population,
    population_description,
    super_population,
    super_population_description,
    COUNT(population) AS population_count,
  FROM
    `genomics-public-data.1000_genomes.sample_info`
  WHERE
    In_Phase1_Integrated_Variant_Set = TRUE
  GROUP BY
    population,
    population_description,
    super_population,
    super_population_description),(select count(population) as total_count from  `genomics-public-data.1000_genomes.sample_info` where In_Phase1_Integrated_Variant_Set = TRUE) order by population_ratio
    """

bq_assistant.estimate_query_size(QUERY2) 

df2 = bq_assistant.query_to_pandas_safe(QUERY2)
df2

colours = {"EUR": "blue", "AMR": "red","AFR":"orange","EAS":"green"}

from matplotlib.patches import Patch
#plt.xticks(df2["population_count"], df2["population"])
plt.figure(figsize=(16, 8))
plt.xlabel('Populations')
plt.ylabel('Count of samples in population')
test_df = df2.sort_values(by=['population'])


#colors = tuple(np.where(test_df["super_population"] =="EAS", 'green','red') )
plt.bar(test_df["population"],test_df["population_count"], color=test_df["super_population"].replace(colours))
plt.legend(
    [
        Patch(facecolor=colours['EUR']),
        Patch(facecolor=colours['AMR']),
        Patch(facecolor=colours['AFR']),
        Patch(facecolor=colours['EAS'])
    ], ["EUR", "AMR","AFR","EAS"]
) 
plt.show()

QUERY3= """
SELECT
  super_population,
  super_population_description,
  super_population_count,
   (super_population_count/SUM(total_count) OVER (PARTITION BY super_population)) AS super_population_ratio
from(
  SELECT
    super_population,
    super_population_description,
    COUNT(population) AS super_population_count,
  FROM
    `genomics-public-data.1000_genomes.sample_info`
  WHERE
    In_Phase1_Integrated_Variant_Set = TRUE
  GROUP BY
    super_population,
    super_population_description),(select count(super_population) as total_count from  `genomics-public-data.1000_genomes.sample_info` where In_Phase1_Integrated_Variant_Set = TRUE)  ORDER BY super_population_ratio
    """

bq_assistant.estimate_query_size(QUERY3) 

df3 = bq_assistant.query_to_pandas_safe(QUERY3)
df3

from matplotlib.patches import Patch
#plt.xticks(df2["population_count"], df2["population"])
plt.figure(figsize=(6, 6))

plt.ylabel('super population count')
test_df = df3.sort_values(by=['super_population'])


#colors = tuple(np.where(test_df["super_population"] =="EAS", 'green','red') )
plt.bar(test_df["super_population"],test_df["super_population_count"], color=test_df["super_population"].replace(colours))
plt.legend(
    [
        Patch(facecolor=colours['EUR']),
        Patch(facecolor=colours['AMR']),
        Patch(facecolor=colours['AFR']),
        Patch(facecolor=colours['EAS'])
    ], ["EUR", "AMR","AFR","EAS"]
) 
plt.show()

QUERY4= """
SELECT
  population,
  gender,
  population_count,
  (population_count/SUM(total_count) OVER (PARTITION BY gender))
  AS population_ratio
from(
  SELECT
    gender,
    population,
    COUNT(population) AS population_count,
  FROM
    `genomics-public-data.1000_genomes.sample_info`
  WHERE
    In_Phase1_Integrated_Variant_Set = TRUE
  GROUP BY
    gender,
    population)
,(select count(population) as total_count from  `genomics-public-data.1000_genomes.sample_info` where In_Phase1_Integrated_Variant_Set = TRUE)  ORDER BY population, gender
    """

bq_assistant.estimate_query_size(QUERY4) 

df4 = bq_assistant.query_to_pandas_safe(QUERY4)
df4

female_df=df4.query('gender=="female"')
female_df

male_df=df4.query('gender=="male"')
male_df




fig, ax = plt.subplots()
index =np.arange(14)
bar_width = 0.35
opacity =0.8

rects1 = plt.bar(index, male_df["population_count"], bar_width,
alpha=opacity,
color='g',
label='Men')

rects2 = plt.bar(index + bar_width, female_df["population_count"], bar_width,
alpha=opacity,
color='r',
label='Women')

plt.xlabel('Population')
plt.ylabel('count of samples in population')

plt.xticks(index + bar_width, ('ASW','CEU','CHB','CHS','CLM','FIN','GBR','IBS','JPT','LWK','MXL','PUR','TSI','YRI'))
plt.legend()

#plt.tight_layout()
plt.show()

QUERY5= """
SELECT
num_family_members AS family_size,
COUNT(num_family_members) AS num_families_of_size
FROM (
  SELECT
  family_id,
  COUNT(family_id) AS num_family_members,
  FROM
  `genomics-public-data.1000_genomes.sample_info`
  WHERE
  In_Phase1_Integrated_Variant_Set = TRUE
  GROUP BY
  family_id)
GROUP BY
family_size
    """

bq_assistant.estimate_query_size(QUERY5) 

df5 = bq_assistant.query_to_pandas_safe(QUERY5)
df5

import matplotlib.pyplot as plt
plt.figure(figsize=(5, 6))
plt.plot(df5["family_size"],df5["num_families_of_size"])
plt.xlabel('Number of family members')
plt.ylabel('Count of families of size')
plt.grid(True)
plt.show()