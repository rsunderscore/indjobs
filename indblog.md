# Classification of Job postings
## Objective

I don't get enought data analysis it my day-to-day work so I do these hobby projects on the side to hone my skills and learn new ones.  If one is unsatisfied with the amount of analysis in their job the goal would be to find a new.  How can we leverage machine learning and python in order to make the job hunt easier. I plan to start the work and extend it indefinitely as long as my interest holds.  So I will start with basic data gathering, move on to machine learning, and might touch on some neural networks at some point in the future.  The current plan is to revise this blog post as I go rather than creating new ones, so that the final product will be a single relatively complete blog post on the topic. 


## Requirements gathering
In this case I am the custoemr so I can define the requirements and work toward them.
- advanced filtering - block certain companies or keywords from ever appearing
    - weight some search terms as more important than others
- rank jobs based on how closely they match my resume as closely as possible
    - point out close, but not exact matches so the terminology in teh resume can be modified to match the job post (custom synonyms/stems based on jargon)
- put jobs into groups based on their requirements (clustering)
- rank jobs based on how similar they are to other jobs we've applied for (classification)
- recommend jobs to apply for (binary classification)
- summarize job postings based on key terms (topic)

## Data gathering
It would be nice to pull a lot of job postings to use as the corpus for the model. Indeed has some sparse API documentation. Most of their documentation is focused on people creating web-apps where they would allow a visiting user to authenticate to their Indeed account using Oauth tokens. In my case, I just want to pull a lot of data for myself, not act on behalf of other users. (I ran into this same problem when I tried to research API calls for LinkedIn.) There was a nice blog on medium.com that talks about pulling job search results from Indeed API, and mentions that a publisher ID is required. The publisher documentation is even more sparse and the difficulty in finding it from the indeed main site made me suspicious. I put some relatively generic information in their form to apply for this access and received no response so the API was a non-starter.  
Navigating the Indeed site I noticed they were passing get paramaters for most of the pages, which would be easy enough to replicate. I played around with the url until I foudn the key components that were needed to make a worthwhile search:
- `q=text` this is the keywords you enter for the search
- `l=text` this is the location you enter to search around
- `radius=int` the number of miles within which to search
- `start=int` ressults are paginated at 10 items so this param allows access to more results (not needed for first page)

As with most modern web pages there is an immense sea of custom scripting, CSS, and meta tags to search through to find valid data elements.  In an ideal world, results would be in table and we could just use pandas `read_html` to get the appropriate content.  As of this writing there were no tables in the html output and everything was in custom tagged div elements.  They were using a construct called jobCard to show summary information and a truncated description ont he left.  The whole card acts as a link that triggers detail data to appear to the right, in an inline frame.  The link shown for the jobCard was very complicated, with params like: mo, ad, vjs, jsa, tk.  The ad param appeared to be a hash of several hundred characters, perhaps a session id, or encrypted data.  Rendering the iframe results in a new page yielded a much simpler url pointing to a viewjob page with easier to parse parameters:
- `jk=int(16)` - this appeared to be a hash that identified the job posting
- `viewtype=embeeded` - this param was used to narrow the formatting so the page would fit when rendered in the inline frame
- `tk=int(16)` - this also appeared to be an identifier, but did not change with each job (maybe a session id)

Through experimentation jk was identified as the only paramter that was needed to get job details.  The job title was readily available in the only h1 tag on tha page but other elements were in nested div tags with custom multi-value class attributes.  BeautifulSoup find methods with a `class_` filter will earch among all of teh listed classes for an element so they were able to find some useful tags:
- CompanyInfo - name, address, review rating - there were various classes matching this string and depending on the nature of the company it had a few variations - e.g. CompanyInfoWithoutHeaderImage, CompanyInfoWithReview
- JobComponent-description - this was the class used to identify the job description text
- there was a tag with id='jobDescriptionText' which also contained the job description but the role type was sometimes oustide of this tag

## automation of data gathering
I was able to extract the job detail links with a regular expression and found the results correlated with the content on sampled pages: `(?:\")[\w/]*?viewjob\?[\w=&]+(?P<jk>jk=\w+)[\w=&-]+.{1,3}`.  I wrote functions to extract on these links and visit them with `requests` to download their contents.  The html returned for each job was parsed for the classes that appeared to contain the details we needed.  This functionality was put into a python file that could be loaded to jupyter notebooks as a module.  I started to put the functionality into a class named `indweb` to encapsulate the scraping component but this is not yet complete.  


I was worried that there would be many dynamic calls to render the results but upon examination of the returned URL it appeared that most key data was present. The map field was rendered after the fact and was not accessible, but similar data could be found elsewhere on the page.  I used BeautifulSoup again to parse the HTML and remove some unneeded tags with this utility function.
```python    
def clean_soup(soup_or_html, rem_tags = ['meta', 'script', 'style', 'svg', 'button'], ch_tags = ['div','span']):
    """
    soup: BeautifulSoup object (bs4)
    rem_tags: tags to remove completely using 'extract' (e.g. script, style, meta)
    ch_tags: tags to replace with their children (e.g. div, span)
    """
    #make a copy so orig stays intact
    #str of soup is str and str of str is str so don't need a branch
    #if isinstance(s3, BeautifulSoup):
    soupcopy = BeautifulSoup(str(soup_or_html))

    for t in rem_tags:
        for e in soupcopy.find_all(t):
            e.extract()
    for t in ch_tags:
        for x in soupcopy.find_all(t):
            x.replace_with_children()
    return soupcopy
```
I felt obligated to make a copy of the original soup object so that modification of the original object would be explicit, if desired. (zen of python rule #2)
Search results were contained in individual table elements with a class string that contained 'JobCard'. Unfortunately there were two variants of this class label, one with 'ShelfContainer' and another with '_mainContent'; so a simple find_all would not be able to locate all of the relevant tables. _mainContent tables contained information about the employer and ShelfContainer tables contained the actual job details. 
I found it convenient to view tag attributes from find_all by passing them to a pandas DataFrame, to automatically handle any tags that were inconsistent (or missing) between the resulting tag elements.
`pd.DataFrame([x.attrs for x  in s2.find('body').find_all('table')])`

---
sidebar
>There was also a mysterious 'ts' element that would show up in the url parameters. Visual comparison to a datetime timestamp revealed a striking similarity, but the ts was an integer with too many digits. Datetime timestamps are a float with only 10 digits in the whole number part. I wrote another utility function to convert integers longer than 10 digits to a datetime compatible float for conversion. This function yielded reasonable dates and times based on my testing. 
```python
def convert_indeed_ts(ts):
    exp = len(str(ts))-10
    res = ts/10**exp
    return datetime.fromtimestamp(res)
 ```
 
## Data Exploration (EDA)
 
## Topic analysis
 
## similarity of documents
- cosine simliarity

- other methods
 
 ## clustering
 
 ## classification
 If we want to consider it as a classification problem we can use whether we applied or not as the target variable.  
 ## bayesian classification
