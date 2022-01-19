#place to put utility functions
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime
import re, os
import requests
import traceback
from urllib.parse import quote
import sqlalchemy as sa


## create a DB to hold data that we need

DBLOC = r'ignoredir\indjobs.sqlite.db'


def create_db(dbloc):
    import sqlite3
    conn = sqlite3.connect(dbloc)
    print(sqlite3.version)
    conn.close()
    print(f"created db with file {dbloc}")
    
if not os.path.exists(DBLOC) & os.path.isfile(DBLOC):
    print(f"db not found at {DBLOC} ... creating")
    create_db(DBLOC)


sldb = sa.create_engine('sqlite:///'+DBLOC)
print('db ready')


class indweb():
    from urllib.parse import quote, unquote
    import re
    
    jobdetail_ex_redirect = r'https://apply.indeed.com/indeedapply/confirmemail/viewjob?iaUid=1fk69eiv0t5i5803&apiToken=aa102235a5ccb18bd3668c0e14aa3ea7e2503cfac2a7a9bf3d6549899e125af4&next=https%3A%2F%2Fwww.indeed.com%2Fviewjob%3Fjk%3D320881414fe4590c'
    jobdetail_ex = r'https://www.indeed.com/viewjob?jk=320881414fe4590c'
    jobdetail_url = r'https://www.indeed.com/viewjob?jk='
    jobsrch_url=r'https://www.indeed.com/jobs?q=querytext&l=location&radius=5'
    jobsrch_url_ext = r'https://www.indeed.com/jobs?q=querytext&radius=5&l=location&ts=timestamp&fromage=last'#&rq=1&rsIdx=3&vjk=291baaf89bfb45be&newcount=141
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



### things that could potentially move into the class
def get_url_jk_part(urlpart):
    """given a url substring containing the jk id from the job detail page - strip out the jk id and return it
    string urlpart: a url substring containing the jk id
    returns (str): the jk id param only e.g. jk=886c10571b6df72a which can be appended to a working job detail url
    
    """
    jkrx= re.compile(r'jk=(\w+)', re.I)
    m=jkrx.search(urlpart)
    if not m:
        print('could not find jk part of url')
        return None
    #print('matched string is ', m.groups()[0])
    jkpart = m.groups()[0]
    
    return jkpart

def get_jobdet_page(urlpart,urlonly = False):
    """given a url - retrieve the job detail html from the web
    strips the jk param from the provided url and feeds it to the working url for job detail information
    urlpart: any part of url containing the jk id for the job detail page e.g. https://www.indeed.com/viewjob?jk=886c10571b6df72a
    urlonly: return None and print the url created by stripping out the jk id and appending to working link
    
    return: a response object or None on bad status code
    """

    jkpart = get_url_jk_part(urlpart)
    if not jkpart:
        print('could not find the jk id in the url')
        return None

    jobdet_url = indweb.jobdetail_url+jkpart
    print(f'url for job detail is {jobdet_url}')
    if urlonly:
        return None
    resp = requests.get(jobdet_url)
    if resp.status_code != 200:
        print(b'bad status code when retrieving html: {resp.status_code}')
        return None

    return resp

def get_job_detail(html, descrtype='full'):
    """parse the html (i.e. resp.content) for relevant job detail information (e.g. company, description, etc...)
    str html: some html to parse using bs4.BeautifulSoup
    str descrtype ['min', 'full']: there are 2 div containers, 'full' option uses the one that gives slightly more information at the top (full/part time and salary est)
    returns (dict): returns a dict of found fields or None on error
        dict keys are title, company, description, address
            address is frequntly null because it comes from a part of the page that is loaded dynamically, the company field typically contains the addr if available
    """
    company_classes = ['jobsearch-CompanyInfoWithoutHeaderImage','jobsearch-CompanyInfoWithReview']
    if not descrtype in ['min', 'full']:
        print(f'descrtype must be min or full, not {descrtype}')
        return None
    
    detsoup = BeautifulSoup(html, features='lxml')
    #adding whitespace to prevent strings running together when extracting text from elements
    psoup=BeautifulSoup(detsoup.prettify(), features='lxml') 
    b = psoup.find('body')
    try:
        title =b.find('h1').text.strip()#class should contain jobsearch-JobInfoHeader-title #.attrs['class']
        #company might be under jobsearch-CompanyInfoWithoutHeaderImage class
        c = b.find('div', class_ = 'jobsearch-CompanyInfoWithReview')
        if not c:
            c= b.find('div', class_='jobsearch-CompanyInfoWithoutHeaderImage')
        company = indweb.condense_whitespace(c.text.strip())

        if descrtype  == 'full':# has a few more fields e.g. seasonal job type
            description = indweb.condense_whitespace(b.find(class_="jobsearch-JobComponent-description").text)
        else:
            description = indweb.condense_whitespace(b.find(id='jobDescriptionText').text)

        #it looks like map address is loaded via scripts after intial html response is provided - 
        # is address always part of the company field when available?
        address = b.find('div', id='mapRoot').text
    except AttributeError as e:
        print(type(e), e)
        traceback.print_tb(e.__traceback__)
        #print(sys.exc_info())  #most internest searches return this and following line
        #traceback.print_exc() # this works
        return b
    
    return {'title':title, 'company':company, 'description':description, 'address':address}

def process_jobdet_url(url):
    """ given a url - parse out the relevant part to get a job detail page and return a dataframe of key fields
    url: this can be any form of url as long as it contains the jk=90u8j0ip3med990x sixteen(16) character id
    
    """
    resp = get_jobdet_page(url)
    if not resp:
        print('there was a problem getting the url')
        return None
    resdict = get_job_detail(resp.content, 'full')
    if not isinstance(resdict, dict):
        print('there was a problem - returning soup object for html body')
        return resdict
    #no problems
    resdict['jk'] = resp.url
    df = pd.DataFrame(resdict, index=[0])
    
    return df

def get_jobdet_loop(url_iter, pause=5):
    """ loop through urls and request the job detail page, parse out fields and save to db
    url_iter(iter): any iterable containing url substrings that have the jk id
    pause(int): how long to wait between http requests to avoid (appearance of) flooding the site
    returns(pd.DataFrame): after saving all the jobs to the db a select is run for all saved jobs and results are returned
    """
    from time import sleep
    for l in url_iter:
        print(f"processing {l} ... ", end="")
        df = process_jobdet_url(l)
        if isinstance(df, pd.DataFrame):
            df.to_sql('job_postings',  sldb, if_exists='append',index=False,)
        print("done")
        sleep(pause)
    print('all done')
    saved_jobs = pd.read_sql('select * from job_postings', sldb).set_index('index')
    return saved_jobs
   

### utility functions

def test_convert_indeed_ts():
    ts = 1636682493568#1636493555554
    return convert_indeed_ts(ts).strftime('%x %X')
    

def convert_indeed_ts(ts):
    exp = len(str(ts))-10
    res = ts/10**exp
    return datetime.fromtimestamp(res)

def parse_url_params(url):
    l = url.split('?', 1)
    url, params = [l[0], "?".join(l[1:])] #if one of the params happens to be a url keep it intact
    params = [x.split('=') for x in params.split('&')]
    params = dict([[x[0], "=".join(x[1:])] for x in params]) #make sure only 2 elements per dict entry

    return url, params


def clean_soup(soup_or_html, rem_tags = ['meta', 'script', 'noscript', 'style', 'svg'], ch_tags = ['div','span']):
    """
    soup: BeautifulSoup object (bs4)
    rem_tags: tags to remove completely using 'extract' (e.g. script, style, meta)
    ch_tags: tags to replace with their children (e.g. div, span)

    """
    #make a copy so orig stays intact
    #str of soup is str and str of str is str so don't need a branch
    #if isinstance(s3, BeautifulSoup):
    soupcopy = BeautifulSoup(str(soup_or_html), features='lxml')
    
    for t in rem_tags:
        for e in soupcopy.find_all(t):
            e.extract()
    for t in ch_tags:
        for x in soupcopy.find_all(t):
            x.replace_with_children()
    return soupcopy


### ML utility functions
def do_sil_plot(silhouette_vals, pred):
    """
        do_a silhouette plot based on values retruned from sklearn.metrics.silhouette samples
        - used for clustering algorithms to determine clusters sizes and proximity
        - works better on spherical?
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm #colormaps stuff
    cluster_labels=np.unique(pred)
    n_clusters=len(cluster_labels)
    y_min, y_max = 0,0
    yticks=[]
    for i, c in enumerate(cluster_labels):
        print(i,c)
        c_sil_vals = silhouette_vals[pred == c]
        c_sil_vals.sort()
        y_max += len(c_sil_vals)
        color = cm.jet(float(i)/n_clusters) #set the color based on the current cluster index relative to cluster count
        plt.barh(range(y_min, y_max), c_sil_vals, height=1., edgecolor='none',color=color)
        yticks.append((y_min+y_max)/2.)# set the tick mark to appear halfway up the range of bars
        y_min += len(c_sil_vals)#set the starting point fo rhte next set of bars
    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color='red', linestyle='--')
    plt.yticks(yticks, cluster_labels+1)
    plt.ylabel='Cluster'
    plt.xlabel='Silhouette coefficient'
    plt.tight_layout()
    plt.show()

def get_confusion(CF, i):
    """
    demonstration of taking apart the multi-variate confusion matrix to get the 2x2
    and also calculate the common metrics for each classification group
    CF: a multivariate confusiotn matrix (DataFrame from ndarray)
    i: the variable index for which to show the traditional 2x2 confusion matrix
    
    computed accuracy score matches with the 'macro avg' computed by sklearn.metrics.classification_report
    """
    TP = CF.loc[i,i]
    FN = CF.loc[i].sum() - TP #the row (predictions)
    FP = CF.loc[:, i].sum() - TP #the column (actuals)
    TN = CF.values.sum() - CF.loc[i].sum() - CF.loc[:,i].sum() + TP # or CF.values.sum() - FN - FP
    #since TP is subtracted twice (as part of the row and as part of the column - need to add it back once to get the right total)
    cfout =pd.DataFrame([[TP,FN],[FP,TN]], columns = ['pred_pos', 'pred_neg'], index=['act_pos', 'act_neg'])
    #act_neg = FP + TN
    accuracy = (TP+TN)/(TP+TN+FP+FN) # (TP+TN)/(TP+FN)
    precision = TP / (TP+FP) #yes, but no? #TP/pred_pos #pred_pos = TP + FP # 
    recall =  TP/(TP+FN) #TP/act_pos #act_pos = TP + FN#
    f1 = 2*(precision * recall) / (precision+recall)
    #print((i), f"TP is {TP}; FP is {FP}; FN is {FN}; TN is {TN}" )
    cols = ['TP', 'FP', 'FN', 'TN', 'accuracy', 'precision', 'recall', 'f1']
    res = pd.Series([TP, FP, FN, TN, accuracy, precision, recall, f1 ], index=cols)
    return res

def eudist(p1, p2):
    #sqrt of (x2-x1)**2 + (y2-y1)**2
    return np.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)

def npdist(p1, p2):
    """calculate distance using numpy.linalg.norm
     Euclidean distance is equivalent to the l2 norm of the difference between the two points
     i.e. np.linalg.norm(np.array((10,0))-np.array((1,0)),2) == 9
    """
    return np.linalg.norm(p2-p1, 2)

def test_eudist():
    
    p1 = (0, 0)
    p2 = (0, 0)
    assert eudist(p1,p2) == 0
    
    p2 = (10,0)
    assert eudist(p1,p2) == 10
    
    p2 = (0, 10)
    assert eudist(p1,p2) ==10
    
    p2 = (4, 3)
    assert eudist(p1,p2) ==5

def inersha(pts, ctrs): 
    """ 
    sum of squared distances of points from their nearest centroid
    given points and centroids - compute the inertia (how close each point is to nearest centrooid)
    return the sum
    """
    res = []
    for pt in pts:
        #compute distances from this point to each center an keep the minimum one
        print(f"for point {pt}", end="---")
        mindist = np.inf
        for c in ctrs:
            #commpute distance
            dist = eudist(pt, c)
            if dist < mindist: 
                msg = f"{c} is closest with dist {dist:.2f}"
                mindist = dist
        print(msg)
        res.append(mindist**2)
    return sum(res)

def test_inersha():
    pts = [(0,10),(5,5),(4,3),(1,1),(3,4), (8, 5), (0, 1), (5, 1), (5, 9), (0, 8), (1, 5), (1, 8), (1, 3), (2, 6), (6, 2), [1, 2], [8, 2], [4, 0], [6, 2], [4, 9], [8, 6], [9, 2], [3, 2], [7, 5], [6, 5]]
    ctrs = [(0,0), (0,10), (7,7)]
    inersha(pts, ctrs) == 330
    
def get_all_confusion(CF, target_names):
    allgroups = pd.Series(CF.index).apply(lambda x: get_confusion(CF, x))
    avgofrows = allgroups.mean()
    avgofrows.name='ROW AVG'
    allgroups.index = target_names #replace numeric index with category names
    return pd.concat((allgroups, avgofrows.to_frame().T))

def get_newsgroup_data(uselocalcopy = True, remove = ('headers', 'footers', 'quotes'), categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']):
    import shelve as s
    import os
    shelfloc = r'c:\temp\20newsgroups.dat'
    data_train = None
    data_test = None
    if os.path.exists(shelfloc) and uselocalcopy:
        with s.open(os.path.splitext(shelfloc)[0]) as datastore:
            try: 
                data_train = datastore['data_train']
                data_test = datastore['data_test']
            except KeyError as e:
                print(f'ERROR: could not find key: {e.args[0].decode()}')
                get_newsgroup_data(False)
    else:
        print(f'updating local copy at {shelfloc}')
        from sklearn.datasets import fetch_20newsgroups
        data_train = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42, remove=remove)
        data_test = fetch_20newsgroups(subset="test", categories=categories, shuffle=True, random_state=42, remove=remove)
        
        #save for later use without doing download
        with s.open(os.path.splitext(shelfloc)[0]) as datastore:
            datastore['data_train'] = data_train
            datastore['data_test'] = data_test
    return data_train, data_test

def size_mb(docs):
    return sum(len(s.encode("utf-8")) for s in docs) / 1e6