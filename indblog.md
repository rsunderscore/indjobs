# Classification of Job postings

## Data exploration
It would be nice to pull a lot of job postings to use as the corpus for the model. Indeed has some sparse API documentation. Most of their documentation is focused on people creating web-apps where they would allow a visiting user to authenticate to their Indeed account using Oauth tokens. In my case, I just want to pull a lot of data for myself, not act on behalf of other users. (I ran into this same problem when I tried to research API calls for LinkedIn.) There was a nice blog on medium.com that talks about pulling job search results from Indeed API, and mentions that a publisher ID is required. The publisher documentation is even more sparse and the difficulty in finding it from the indeed main site made me suspicious. In the end I just used very generic credentials so we'll see if that gets a past a review.
Upon further examination I found that navigating the urls for the site leveraged GET parameters, which could be easily replicated with the requests package. I was worried that there would be some dynamic calls to render the results but upon examination of the returned URL it appeared that all the relevant information was present. I used BeautifulSoup again to parse the HTML and remove some unneeded tags with this utility function.
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
