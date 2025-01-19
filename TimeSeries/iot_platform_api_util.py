import json
from urllib.parse import urlparse
from api_util.http_request import http_get, http_post
from api_util.mqtt_request import subscribing, subscribing_once
from api_util.iot_platform_dto.post_cin_dto import M2MPostCinDTO
from api_util.iot_platform_dto.post_cnt_dto import M2MPostCntDTO

def get_iot_platform_cnt_lbl(url):
    cnt_dict = json.loads(http_get(url, iotPlatform = True).text)
    cnt_lbl = cnt_dict['m2m:cnt']["lbl"]

    return cnt_lbl

def get_iot_platform_sub_nu(url):
    sub_dict = json.loads(http_get(url, iotPlatform = True).text)
    sub_nu = sub_dict['m2m:sub']["nu"]

    return sub_nu

def post_iot_platform_sensing_cin(url, sensor_data):
    cin = M2MPostCinDTO(sensor_data)
    response = http_post(url, json=cin.m2m_cin, iotPlatform = True, ty=4)
    return response

def post_iot_platform_cnt(url, cnt_name, lbl = []):
    cnt = M2MPostCntDTO(cnt_name, lbl)
    response = http_post(url, json=cnt.m2m_cnt, iotPlatform = True, ty=3)
    return response

def sub_iot_platform_cin_con(nu, port):
    ip = urlparse(nu).hostname
    path_segments = urlparse(nu).path.split('/')
    topic = next(segment for segment in path_segments if segment)
    print(topic)
    port = int(port)

    received_msg = subscribing_once(ip, port, topic)
    print(received_msg)
    parsed_msg = received_msg['pc']['m2m:sgn']['nev']['rep']['m2m:cin']['con']

    try:
        parsed_msg = json.loads(parsed_msg)
    except:
        pass

    return parsed_msg

if __name__ == "__main__":
    post_url = 'http://203.250.148.120/service408/Sensor1/train/ce7840ca'
    nu = 'mqtt://203.250.148.120/service408Sensor1?ct=json'
    port = 20516
    x = sub_iot_platform_cin_con(nu, port)