# Classification of Job postings
## Objective

I don't get enought data analysis it my day-to-day work so I do these hobby projects on the side to hone my skills and learn new ones.  I'm also a fan of overlapping hobbies: if I want to learn spanish and python I use python on spanish text, if I want to search jobs and learn data science I use python on job data.  Since much of data science is obtaining the data and manipulating it, we begin by examining our data source to determine how to extract data.  Once we have a reasonable dataset we can start using machine learning, and then move on to neural networks.  I'm starting with this initial post and will modify, or add to it, as I go.  To know what to gather we will establish some basic requirements, but allow ourselves to be agile and adjust them over time.  


## Requirements 
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
It would be nice to pull a lot of job postings to use as the corpus for the model. Indeed has some sparse API documentation. Most of their documentation is focused on people creating web-apps where they would allow a visiting user to authenticate to their Indeed account using Oauth tokens. In my case, I just want to pull a lot of data for myself, not act on behalf of other users. (I ran into this same problem when I tried to research API calls for LinkedIn.) There was a nice blog on medium.com that talks about pulling job search results from Indeed API, and mentions that a publisher ID is required. The publisher documentation is quite sparse and difficult to find from the main site, raising suspicion.  I put some relatively generic information in their form to apply for this access and received no response, so the API was a non-starter.  

The next alternative was scraping data from the site directly with `requests` or `scrapy` or similar python tools. I'm not a fan of sending lots of html requests to public sites, but if I metered my connection I assumed this would be a reasonable workaround.  Clicking around the site, I noticed 'get' parameters in the urls that I might be able to leverage.  For the search page, 'jobs', these are the parameters I found most useful.
**job search parameters**
- `q=text` this is the keywords you enter for the search
- `l=text` this is the location you enter to search around
- `radius=int` the number of miles within which to search
- `start=int` results are paginated at 10 items so this param allows access to more results (not needed for first page)

They were using a construct called jobCard to show summary information and a truncated description on the left.  The whole card acts as a link that triggers detail data to appear to the right, in an inline frame.  The link shown for the jobCard was long and complicated, with params like: mo, ad, vjs, jsa, tk. Most of the length was the ad param, which contained several hundred characters and appeared to be a hash of some kind.  A right click on the iframe result yielded the option to render in a new tab, and the results were suing a much simpler url and parameters. 
**job detail parameters**
- `jk=int(16)` - this appeared to be a hash that identified the job posting
- `viewtype=embeeded` - this param was used to narrow the formatting so the page would fit when rendered in the inline frame
- `tk=int(16)` - this also appeared to be an identifier, but did not change with each job (maybe a session id)

I tried various combinations of parameter values and found that `jk` was the only item needed to obtain job details.  

## getting data
As with most modern web pages there is an immense sea of custom scripting, CSS, and meta tags to search through to find valid data elements. I had hoped results would be in table and we could just use pandas `read_html` to get the appropriate content, but this was not the case.  So we need to parse the html and look for distinguishing features and patterns, usually with python's `BeautifulSoup` (bs4) library.  Since this paradigm comes up so often, I wrote a custom function to do some clean up on the html to make things easier to view.
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

I was able to locate the urls for the iframe in script tags contained in each jobCard. Each link could be extracted with a regular expression, `(?:\")[\w/]*?viewjob\?[\w=&]+(?P<jk>jk=\w+)[\w=&-]+.{1,3}`.  I wrote functions to extract all of the matching links and download their contents.  The html returned for each job was parsed for the classes that appeared to contain the details we needed.  This functionality was put into a python file that could be loaded to jupyter notebooks as a module.  I started to put the functionality into a class named `indweb` to encapsulate the scraping component but this is not yet complete.

```python
class indweb():
    from urllib.parse import quote, unquote
    import re
    

    jobdetail_url = r'https://www.indeed.com/viewjob?jk='
    jobsrch_url=r'https://www.indeed.com/jobs?q=querytext&l=location&radius=5'
    joblink_regex = re.compile(r'(?:\")[\w/]*?viewjob\?[\w=&]+(?P<jk>jk=\w+)[\w=&-]+.{1,3}',re.IGNORECASE|re.UNICODE)
    def __init__(self):
        self.url = ""
        
        #attrs that are only populated after a method is run
        self.response_ = None
        self.html_ = None
        self.joblinks_ = None
        
    def __str__(self):
        s=""
        for k in self.__dict__:
            s += f"{k} has value {self.__dict__[k]}\n"
        return s
    
    def fmt_jobsrch_url(self, keyword, location):
        return self.jobsrch_url.replace('querytext',quote(keyword)).replace('location', quote(location))
    
    @staticmethod
    def parse_url_params(url):
        l = url.split('?', 1)
        url, params = [l[0], "?".join(l[1:])] #if one of the params happens to be a url keep it intact
        params = [x.split('=') for x in params.split('&')]
        params = dict([[x[0], "=".join(x[1:])] for x in params]) #make sure only 2 elements per dict entry

        return url, params
    
    def get_jobsrch_page(self, start = 0, keywords='python', loc='12345'):
        search_url_works=self.fmt_jobsrch_url(keywords, loc)
        if start > 0: 
            search_url_works+=f"&start={start}"
        print(search_url_works.replace(loc,'xxxxx'))
        js_resp = requests.get(search_url_works)
        if js_resp.status_code !=200:
            print(f"bad status for response {js_resp.status_code}")
            return None
        else:
            self.response = js_resp
            self.html_ = js_resp.content.decode()
            return js_resp
        
    def get_joblinks_from_html(self, html=None):
        if not html: html = self.html_
        rx = re.compile(r'(?:\")[\w/]*?viewjob\?[\w=&]+(?P<jk>jk=\w+)[\w=&-]+.{1,3}',re.IGNORECASE|re.UNICODE)
        res = rx.finditer(html)# findall would return just the paren match string m.groups()[0]
        self.joblinks_ = res
        return res
    
    @staticmethod
    def condense_whitespace(text):
        return re.sub(r'\s{2,}','\n',text.strip())
```

The key fields we would use were within the html for the job details of each link. The job title was readily available in the only 'h1' tag on tha page but other elements were in nested 'div' tags with custom multi-value class attributes.  BeautifulSoup `find` methods with a `class_` filter will earch among all of teh listed classes for an element so they were able to find some useful tags:
- CompanyInfo - name, address, review rating - there were various classes matching this string and depending on the nature of the company it had a few variations - e.g. CompanyInfoWithoutHeaderImage, CompanyInfoWithReview
- JobComponent-description - this was the class used to identify the job description text
- there was a tag with id='jobDescriptionText' which also contained the job description but the role type was sometimes oustide of this tag

I had some initial concern about dynamic rendering of the content resulting in data gaps, but other than the map field (which contained an address that was also typically available in the CompanyInfo) I did not have any problems.


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
