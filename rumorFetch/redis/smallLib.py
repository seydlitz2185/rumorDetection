import redis as redis
import requests
import random
import numpy as np
import re
import os
import selenium.webdriver as webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait #等待页面加载某些元素
from selenium.webdriver.support import expected_conditions as EC
import json

pool = redis.ConnectionPool(host='127.0.0.1',port=6379,db = 0,password='123456',decode_responses=True)   

global false, null, true,headers
false = False
null = ''
true = True
headers = {

    'naive_header':{
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'cookie':'ttwid=1|kUxzTmMLz56Qi8ZFSM5oAeco3xt7h1G3Zm2pTOvDJ94|1659753544|9e46280257ebea6ea7a75bd54e1d3a1eb8275a6f7e9573873adff473d1b43729; passport_csrf_token=24e56b98176da2b1a4c52f20dcead20f; passport_csrf_token_default=24e56b98176da2b1a4c52f20dcead20f; s_v_web_id=verify_ld9ofrql_Pfq7wVYw_Qfwb_41oO_9yYl_Kig02zoz95zE; d_ticket=bfbbe2b5b311a303e3b0b9fc4531acf1a46f6; passport_assist_user=CkBfDz10QZodaFeqRVXfecKPWFRvr9Qa1rx16Sqse_8g5fDfmTc-Cwzf-l9Qgq3ru33DFkFCS9UkwiQEeidhp_WBGkgKPFfyb8Wm9xXCNIYMf4mYfvDgh0xMEFG_E6ji7ewxf4CMmZv5rYHpcPXcKRWwzr7niYISxBdcDfsenUmPExCFkqgNGImv1lQiAQPPwis_; n_mh=jszYg9i6gHfZTX7KEHEStvXb1SevpLati_lZm13_1Rk; sso_auth_status=f1958e92d541e4ebef85832766ad1631; sso_auth_status_ss=f1958e92d541e4ebef85832766ad1631; sso_uid_tt=54b57d6b753199fff4e55e2bbdf3cda1; sso_uid_tt_ss=54b57d6b753199fff4e55e2bbdf3cda1; toutiao_sso_user=15f9d447050d2a394a5a1667270cdf25; toutiao_sso_user_ss=15f9d447050d2a394a5a1667270cdf25; sid_ucp_sso_v1=1.0.0-KGFkMTNlNDk0MWE3OWFjZWVlOTMxMDE3YWJmMmZmOTNlZTQ1ZDlmYzIKHwjHueDa1vTiARCgiOmeBhjvMSAMMITr2uoFOAJA8QcaAmxmIiAxNWY5ZDQ0NzA1MGQyYTM5NGE1YTE2NjcyNzBjZGYyNQ; ssid_ucp_sso_v1=1.0.0-KGFkMTNlNDk0MWE3OWFjZWVlOTMxMDE3YWJmMmZmOTNlZTQ1ZDlmYzIKHwjHueDa1vTiARCgiOmeBhjvMSAMMITr2uoFOAJA8QcaAmxmIiAxNWY5ZDQ0NzA1MGQyYTM5NGE1YTE2NjcyNzBjZGYyNQ; odin_tt=6e547e2649e8d112a288ac5b4b94f34637ac2aa0f648a2405b1a7e05e6c83eac44ae4dd7293ffe74c5b6d74fffd9ee2ea974324239f7832df3de0477d077c4a4; passport_auth_status=ef26c08ff0b79dda9440411b19a81cdb,41001d1b7c3773f670ad7000e8b25f8f; passport_auth_status_ss=ef26c08ff0b79dda9440411b19a81cdb,41001d1b7c3773f670ad7000e8b25f8f; uid_tt=37574663df5bc0bfc0213d24267c3a38; uid_tt_ss=37574663df5bc0bfc0213d24267c3a38; sid_tt=932ec4b8f5759cf25124cd0bf93928db; sessionid=932ec4b8f5759cf25124cd0bf93928db; sessionid_ss=932ec4b8f5759cf25124cd0bf93928db; store-region=cn-ln; store-region-src=uid; sid_guard=932ec4b8f5759cf25124cd0bf93928db|1675248676|5183996|Sun,+02-Apr-2023+10:51:12+GMT; sid_ucp_v1=1.0.0-KDhjOTE5MWM0NmY0MzdjY2Y3ZjhlYjMwMTUxZDRhMjc1ZDliMjYyNzYKGQjHueDa1vTiARCkiOmeBhjvMSAMOAJA8QcaAmxmIiA5MzJlYzRiOGY1NzU5Y2YyNTEyNGNkMGJmOTM5MjhkYg; ssid_ucp_v1=1.0.0-KDhjOTE5MWM0NmY0MzdjY2Y3ZjhlYjMwMTUxZDRhMjc1ZDliMjYyNzYKGQjHueDa1vTiARCkiOmeBhjvMSAMOAJA8QcaAmxmIiA5MzJlYzRiOGY1NzU5Y2YyNTEyNGNkMGJmOTM5MjhkYg; download_guide="3/20230201"; live_can_add_dy_2_desktop="0"; _tea_utm_cache_1243=undefined; MONITOR_WEB_ID=9f84d3bf-23cb-453f-9407-9991f6c86282; SEARCH_RESULT_LIST_TYPE="multi"; csrf_session_id=74f75056badb49231401844ec5219f1c; __ac_nonce=063de5a000009e734d347; __ac_signature=_02B4Z6wo00f01C4e7mgAAIDDenObo9vqLfguPurAAGhgVu0SyGuRO4vyoEPe2MdDFf3xOxtvBCwHYuZKnAbjL.sHH5OuOj67G0XjliGZX8arCFMLTl4raVhcimqlNkjxZWx1K7H3fhe9upVIf8; douyin.com; VIDEO_FILTER_MEMO_SELECT={"expireTime":1676121213260,"type":1}; FOLLOW_NUMBER_YELLOW_POINT_INFO="MS4wLjABAAAAqG6QLgkXOH2fY6aUPxhSKMtQ88xZdeuNk49R3m4Hawo/1675526400000/0/1675516413692/0"; strategyABtestKey="1675516415.2"; msToken=kRTYQo_cYbZq2MlCpYFGs70ZZOMuIqfBGBC4u7XoHtzAhYDp5XJK8wwjRzzQuuJILj7Ec48-99u6XoHfnEb1-PJKCO_toDxBdIiEtokPCBmCnAU-KOlgpPgi5Gmw2t0=; tt_scid=B9IrjGIswQ9AT6m6fbJunylXtAZ9WP1A0IK9wwZjhXwfuVLFk6sikbJebdauOtg8e623; msToken=78-NLJMFcE8C-LCx2YSXmRlwcvEpaztojIkogYk2ETcqagJcppJ1R6N6E0RIpoT_ELxXhLKTtLSPzh4IrLO7lvi3id4pPqdr7da9on-lyuwKTkFxPn3IgUbxQS9NnnY=; home_can_add_dy_2_desktop="0"; passport_fe_beating_status=false',
    }
}

chrome_options =  webdriver.ChromeOptions()
    #这一行很关键，让selenium不加载图片，专心爬
chrome_options.add_argument('blink-settings=imagesEnabled=false')
'''
# 设置代理
proxy = get_https_proxy().get('proxy')

chrome_options.add_argument("-proxy-server=https://{}".format(proxy))
chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)')

print(chrome_options.arguments)
'''
preferences = {
    "webrtc.ip_handling_policy": "disable_non_proxied_udp",
    "webrtc.multiple_routes_enabled": False,
    "webrtc.nonproxied_udp_enabled": False
}
chrome_options.add_experimental_option("prefs", preferences)
def get_proxy():
    r = redis.Redis(connection_pool=pool)
    proxy_keys= list( r.hkeys(name='use_proxy'))
    this_key =random.choice(proxy_keys)
    #js = eval(r.hget(name='use_proxy',key=this_key).decode('utf8'))
    return this_key

def get_https_proxy():
    r = redis.Redis(connection_pool=pool)
    https_proxy = []
    while len(https_proxy)==0 :
        proxy_vals= list(r.hvals(name='use_proxy'))
        for val in proxy_vals:
            js = eval(val.decode('utf8'))
            if js.get('https') :
                https_proxy.append(val)
    this_val =random.choice(https_proxy)        
    js = eval(this_val.decode('utf8'))
    return js

def delete_proxy(proxy):
    r = redis.Redis(connection_pool=pool)
    r.hdel('use_proxy',proxy)
    
def getVideo(url):
    # ....
    #url = 'https://'+url
    retry_count = 5
    #http代理似乎已经够用
    proxy = get_proxy()
    #proxy = get_https_proxy().get('proxy')
    #print(proxy)
    while retry_count > 0:
        try:
           # html = requests.get('http://demo.spiderpy.cn/',headers = headers.get('naive_header'),timeout=3,proxies={"http": "http://{}".format(proxy)})
            video = requests.get(url='https://'+url, proxies={"http": "http://{}".format(proxy)},timeout=3, ).content
            # 使用代理访问
            #print('bar')
            return video
        except Exception:
            retry_count -= 1
            #print('foo')
    # 删除代理池中无效代理
    #delete_proxy(proxy)
    raise
    
def getHtml(url):
    retry_count = 5
    #http代理似乎已经够用
    proxy = get_proxy()
    #proxy = get_https_proxy().get('proxy')
    #print(proxy)
    while retry_count > 0:
        try:
           # html = requests.get('http://demo.spiderpy.cn/',headers = headers.get('naive_header'),timeout=3,proxies={"http": "http://{}".format(proxy)})
            html = requests.get(url=url,headers = headers.get('naive_header'), proxies={"http": "http://{}".format(proxy)},timeout=1, )
            # 使用代理访问
            #print('bar')
            return html
        except Exception:
            retry_count -= 1
            #print('foo')
    # 删除代理池中无效代理
    #delete_proxy(proxy)
    raise

def downloadVideo (pool,path,res):
    #print(res)
    #寻找视频名称
    #review_fetcher = "#root > div > div.T_foQflM > div > div > div.leftContainer> div > div > div > div > div > div > div > div> p > span > span > span > span > span > span > span"
    r = redis.Redis(connection_pool=pool)
    title = re.findall('<title data-react-helmet="true">(.*)?</title>',res)[0]
    title = re.sub(r'[\\s\\\\/:\\*\\?\\\"<>\\|]','_',title)
    #print('video'+os.sep+path+os.sep+title +'_page.txt')
    if not os.path.exists('video'+os.sep+path+os.sep+"page"):
        os.makedirs('video'+os.sep+path+os.sep+"page")
    with open ('video'+os.sep+path+os.sep+"page"+os.sep+title +'_page.txt', mode = 'w',encoding='utf-8') as f:
        f.write(res)
    f.close()
    #寻找下载链接
    #hint:src":"为开头，再到其后第一个"为止，是一个视频链接
    href = re.findall('src%22%3A%22%2F%2F(.*?)%22',res)[1]
    video_url = requests.utils.unquote(href)
    #print(video_url)
    try :
        video_content = getVideo(video_url)
        r.hset(path+"video",title,video_url)
        with open ('video'+os.sep+path+os.sep+title +'.mp4', mode = 'wb') as f:
            f.write(video_content)
    except Exception:
        raise
   
#非线性地滑动页面
import time

def drop_down(driver):

    curr = 0
    #先加速
    for x in range (9,1000,40):
        pre = curr
        time.sleep(0.1)
        j = x/9
        js = 'document.documentElement.scrollTop = document.documentElement.scrollTop +'+str(j)+';return document.documentElement.scrollTop;'
        curr =driver.execute_script(js)
        if curr == pre:
            break
    #匀速运行指导到达底部        
    retry = 20
    while retry > 0 :
        pre = curr
        time.sleep(0.1)
        js = 'document.documentElement.scrollTop = document.documentElement.scrollTop +'+str(j)+';return document.documentElement.scrollTop;'
        curr =driver.execute_script(js)
        if pre == curr:
            retry -= 1
            time.sleep(0.5)


        
def down_load_video_from_user_homepage(pool,name,url):
    url = url
    response = None
    path = name
    if not os.path.exists('video'+os.sep+path):
        os.makedirs('video'+os.sep+path)
    try :
        response = getHtml(url)
        res = response.text
        downloadVideo(pool,path,res)
    except Exception:
        raise

def down_load_video_selenium(name,url):
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('blink-settings=imagesEnabled=false')
    #chrome_options.add_argument('headless')
    preferences = {
      "webrtc.ip_handling_policy": "disable_non_proxied_udp",
      "webrtc.multiple_routes_enabled": False,
         "webrtc.nonproxied_udp_enabled": False
    }

    path = 'video'+os.sep+name
    if not os.path.exists(path):
        os.makedirs(path)
    driver = webdriver.Chrome(executable_path='C:\Program Files\chromedriver\chromedriver.exe',chrome_options=chrome_options)
    driver.get(url=url)
    try :
        header = '1'  
    except Exception:
        raise       
    


def get_all_video_form_homepage(url): 
    r = redis.Redis(connection_pool=pool)
    #打开url
    driver = webdriver.Chrome(executable_path='C:\Program Files\chromedriver\chromedriver.exe',chrome_options=chrome_options)
    #driver.get("chrome://version/",proxies = proxies)
    driver.get(url=url)
    wait=WebDriverWait(driver,20)
    wait.until(EC.presence_of_element_located((By.CLASS_NAME,'dy-account-close')))
    time.sleep(5)
    #关闭登录弹窗
    a = driver.find_elements(by= By.CLASS_NAME,value="dy-account-close")
    if len(a) >0 :
        a[0].click()
    #获得用户总视频数
    
    #video_num = driver.find_elements(by= By.CSS_SELECTOR, value="#root > div > div > div > div > div > div:nth-child(2) > div:nth-child(1) > div > div > div > div > span:nth-child(2)")[0]
    #print(int(video_num.get_attribute('innerHTML')))
    drop_down(driver=driver)

    
    #driver.execute_script('document.documentElement.scrollTop=0') 

    #获取页面内容
    #hint:先取所有的li标签，再取a标签中的href
    selector = '#root > div > div > div > div > div > div > div > div > div > ul > li > a'
    lis = driver.find_elements(by= By.CSS_SELECTOR, value= selector)
    cnt = 0
    while cnt < 5 & len(lis)== 0 :
        cnt +=1 
        time.sleep(1)
        lis = driver.find_elements(by= By.CSS_SELECTOR, value= selector)
    
    if len(lis) == 0:
        print(url)
        
    href =[]
    print(len(lis))
    for li in lis:
        try:
            href.append(li.get_attribute('href'))
        except Exception:
            continue
    driver.quit()
    print(len(href))

    return href
    #r = redis.Redis(connection_pool=pool)
    #r.hmset('demo_targets1_hrefs', href)
        #down_load_video_from_user_homepage(url)

def get_user(r,table):
    users= list(r.hkeys(name=table))
    if len(users) == 0:
        return [-1,]
    this_key =random.choice(users)
    js = r.hget(name=table,key=this_key)
    return this_key,js

def delete_user(r,table,name):
    r.hdel(table,name)

def redis_export(table):
    pool = redis.ConnectionPool(host='127.0.0.1',port=6379,password='123456',decode_responses=True)  
    r = redis.Redis(connection_pool=pool)
    store_table=table
    
    h = open(store_table+'.json', 'w',encoding='utf-8')

    keys = r.hkeys(store_table)       
    vals = r.hvals(store_table)
    res=[]
    for i in range(len(keys)):
        dic = {keys[i]:vals[i]}
        res.append(dic)

    h.write(json.dumps(res,ensure_ascii=False)+'\n')

#从table中随机取一个键值对，每次取增加一个访问次数，若访问次数大于10就认为下载失败，将该键值对移入该table名对应的失败hash中
def redis_hpop(r,table):

    proxy_keys= list(r.hkeys(name=table))
    this_key =random.choice(proxy_keys)
    cnt = int(r.hget(table,this_key))+1
    r.hset(table,this_key,cnt)
    if cnt <10:
        return this_key
    else:
        
        r.hdel(table,this_key)
        return None

def restore_table(pool,origin,temp): 
    r = redis.Redis(connection_pool=pool)
    table = r.hgetall(origin)
    temp_table = temp
    r.hmset(temp_table, table)

#设置容忍度，若一个用户主页中下载成功的视频不足总视频量的tolerence，则重新开始下载该用户主页

def check_empty_dir_and_update_redis(pool,farther_dir,table,tolerence):
    r = redis.Redis(connection_pool=pool)
    need_retry={}
    for listx in os.listdir(farther_dir):
        num = len(os.listdir(farther_dir+os.sep+listx))
        ls  = r.hget(table,listx)
        if num > len(ls.split(','))*tolerence:
            #print(listx)
            delete_user(r,table,listx)