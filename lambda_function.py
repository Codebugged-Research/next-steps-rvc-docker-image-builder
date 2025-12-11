import boto3
import json
import time

ec2 = boto3.client('ec2', region_name='ap-south-1')
INSTANCE_ID = 'i-0388b0481741e150b'
ELASTIC_IP = '35.154.212.217'

def lambda_handler(event, context):
    action = event.get('action', 'start')
    
    if action == 'start':
        return start_instance()
    elif action == 'stop':
        return stop_instance()
    elif action == 'status':
        return get_status()
    else:
        return {'statusCode': 400, 'body': json.dumps({'error': 'Invalid action'})}

def start_instance():
    try:
        response = ec2.describe_instances(InstanceIds=[INSTANCE_ID])
        state = response['Reservations'][0]['Instances'][0]['State']['Name']
        
        if state == 'running':
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'status': 'already_running',
                    'endpoint': f'http://{ELASTIC_IP}:5000',
                    'public_ip': ELASTIC_IP
                })
            }
        
        ec2.start_instances(InstanceIds=[INSTANCE_ID])
        
        waiter = ec2.get_waiter('instance_running')
        waiter.wait(InstanceIds=[INSTANCE_ID], WaiterConfig={'Delay': 5, 'MaxAttempts': 40})
        
        time.sleep(45)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'started',
                'endpoint': f'http://{ELASTIC_IP}:5000',
                'public_ip': ELASTIC_IP
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def stop_instance():
    try:
        ec2.stop_instances(InstanceIds=[INSTANCE_ID])
        return {
            'statusCode': 200,
            'body': json.dumps({'status': 'stopping'})
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def get_status():
    try:
        response = ec2.describe_instances(InstanceIds=[INSTANCE_ID])
        state = response['Reservations'][0]['Instances'][0]['State']['Name']
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'status': state,
                'public_ip': ELASTIC_IP,
                'endpoint': f'http://{ELASTIC_IP}:5000' if state == 'running' else None
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
