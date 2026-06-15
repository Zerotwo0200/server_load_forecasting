CPU_WARNING = 70
CPU_CRITICAL = 85
CPU_LOW = 25

RAM_WARNING = 75
RAM_CRITICAL = 90

def generate_recommendation(cpu_pred, ram_pred, active_servers=1):
    
    if cpu_pred >= CPU_CRITICAL or ram_pred >= RAM_CRITICAL:
        return {
            "status": "critical",
            "recommendation": "ADD_2_SERVERS",
            "message": "Critical overload expected",
            "priority": "high"
        }

    if cpu_pred >= CPU_WARNING or ram_pred >= RAM_WARNING:
        return {
            "status": "warning",
            "recommendation": "ADD_SERVER",
            "message": "High load expected",
            "priority": "medium"
        }

    if cpu_pred <= CPU_LOW and active_servers > 1:
        return {
            "status": "low",
            "recommendation": "REMOVE_SERVER",
            "message": "Load decrease expected",
            "priority": "low"
        }

    return {
        "status": "stable",
        "recommendation": "NONE",
        "message": "System stable",
        "priority": "none"
    }
