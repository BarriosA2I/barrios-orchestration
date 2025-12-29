#!/usr/bin/env python3
"""
API Gateway Test Script
Tests all endpoints of the Barrios A2I API Gateway
"""

import asyncio
import json
import aiohttp
import sys

BASE_URL = "http://localhost:8088"


async def test_health():
    """Test health endpoint"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/api/health") as response:
            if response.status == 200:
                data = await response.json()
                print(f"[OK] Health: {data['status']}")
                print(f"    Uptime: {data['uptime_seconds']:.1f}s")
                print(f"    Version: {data['version']}")
                return True
            else:
                print(f"[FAIL] Health check failed: {response.status}")
                return False


async def test_chat():
    """Test chat endpoint"""
    async with aiohttp.ClientSession() as session:
        payload = {
            "message": "What services do you offer?",
            "mode": "auto"
        }
        async with session.post(
            f"{BASE_URL}/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                data = await response.json()
                print(f"[OK] Chat Response:")
                print(f"    Model: {data['model']}")
                print(f"    Confidence: {data['confidence']:.2f}")
                print(f"    Latency: {data['latency_ms']:.2f}ms")
                print(f"    Preview: {data['response'][:80]}...")
                return True
            else:
                text = await response.text()
                print(f"[FAIL] Chat failed: {response.status}")
                print(f"    Error: {text[:200]}")
                return False


async def test_metrics():
    """Test metrics endpoint"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/api/metrics") as response:
            if response.status == 200:
                data = await response.json()
                print(f"[OK] Metrics:")
                print(f"    Total requests: {data['requests']['total']}")
                print(f"    Success: {data['requests']['success']}")
                print(f"    Avg latency: {data['latency']['average_ms']:.2f}ms")
                return True
            else:
                print(f"[FAIL] Metrics failed: {response.status}")
                return False


async def main():
    print("\n" + "=" * 60)
    print("BARRIOS A2I API GATEWAY TEST")
    print("=" * 60 + "\n")

    results = []

    print("Testing Health Endpoint...")
    results.append(await test_health())

    print("\nTesting Chat Endpoint...")
    results.append(await test_chat())

    print("\nTesting Metrics Endpoint...")
    results.append(await test_metrics())

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"RESULTS: {passed}/{total} tests passed")

    if all(results):
        print("[SUCCESS] All tests passed!")
        print("=" * 60 + "\n")
        return 0
    else:
        print("[FAILED] Some tests failed")
        print("=" * 60 + "\n")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except aiohttp.ClientConnectorError:
        print("[ERROR] Cannot connect to API Gateway")
        print("Make sure the server is running: python api_gateway.py")
        sys.exit(1)
