import boto3


def getAllContent():
    """Read database and return a list of content"""
    try: 
        dynamodb = boto3.resource("dynamodb", region_name="ap-southeast-2")
        table_name = "EdgeTable"
        table = dynamodb.Table(table_name)
        
        items = []
        last_evaluated_key = None

        while True:
            # Prepare scan parameters
            scan_params = {}
            if last_evaluated_key:
                scan_params["ExclusiveStartKey"] = last_evaluated_key

            # Scan table
            response = table.scan(**scan_params)

            # Append items
            items.extend(response.get("Items", []))

            # Check for pagination
            last_evaluated_key = response.get("LastEvaluatedKey")
            if not last_evaluated_key:
                break  # No more items to fetch

        return items
        
        
        # response = table.scan()

        # return response["Items"]
    except:
        return []
