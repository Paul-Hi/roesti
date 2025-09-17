#pragma once

#include "macros.cuh"
#include <cuda_runtime.h>
#include <cmath>

namespace roesti
{
    namespace utils
    {
        // #################################################################### float2 operators ####################################################################

        HD inline float2 make_float2(float s)
        {
            return ::make_float2(s, s);
        }

        HD constexpr inline float2 operator+(const float2& a, const float2& b)
        {
            return ::make_float2(a.x + b.x, a.y + b.y);
        }

        HD constexpr inline float2 operator-(const float2& a, const float2& b)
        {
            return ::make_float2(a.x - b.x, a.y - b.y);
        }

        HD constexpr inline float2 operator*(const float2& a, const float2& b)
        {
            return ::make_float2(a.x * b.x, a.y * b.y);
        }

        HD constexpr inline float2 operator/(const float2& a, const float2& b)
        {
            return ::make_float2(a.x / b.x, a.y / b.y);
        }

        HD constexpr inline float2 operator*(const float2& a, float b)
        {
            return ::make_float2(a.x * b, a.y * b);
        }

        HD constexpr inline float2 operator*(float a, const float2& b)
        {
            return ::make_float2(a * b.x, a * b.y);
        }

        HD constexpr inline float2 operator/(const float2& a, float b)
        {
            return ::make_float2(a.x / b, a.y / b);
        }

        HD constexpr inline float2& operator+=(float2& a, const float2& b)
        {
            a.x += b.x;
            a.y += b.y;
            return a;
        }

        HD constexpr inline float2& operator-=(float2& a, const float2& b)
        {
            a.x -= b.x;
            a.y -= b.y;
            return a;
        }

        HD constexpr inline float2& operator*=(float2& a, const float2& b)
        {
            a.x *= b.x;
            a.y *= b.y;
            return a;
        }

        HD constexpr inline float2& operator/=(float2& a, const float2& b)
        {
            a.x /= b.x;
            a.y /= b.y;
            return a;
        }

        HD constexpr inline float2& operator*=(float2& a, float b)
        {
            a.x *= b;
            a.y *= b;
            return a;
        }

        HD constexpr inline float2& operator/=(float2& a, float b)
        {
            a.x /= b;
            a.y /= b;
            return a;
        }

        // #################################################################### float3 operators ####################################################################

        HD constexpr inline float3 make_float3(float s)
        {
            return ::make_float3(s, s, s);
        }

        HD constexpr inline float3 operator+(const float3& a, const float3& b)
        {
            return ::make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
        }

        HD constexpr inline float3 operator-(const float3& a, const float3& b)
        {
            return ::make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
        }

        HD constexpr inline float3 operator*(const float3& a, const float3& b)
        {
            return ::make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
        }

        HD constexpr inline float3 operator/(const float3& a, const float3& b)
        {
            return ::make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
        }

        HD constexpr inline float3 operator*(const float3& a, float b)
        {
            return ::make_float3(a.x * b, a.y * b, a.z * b);
        }

        HD constexpr inline float3 operator*(float a, const float3& b)
        {
            return ::make_float3(a * b.x, a * b.y, a * b.z);
        }

        HD constexpr inline float3 operator/(const float3& a, float b)
        {
            return ::make_float3(a.x / b, a.y / b, a.z / b);
        }

        HD constexpr inline float3& operator+=(float3& a, const float3& b)
        {
            a.x += b.x;
            a.y += b.y;
            a.z += b.z;
            return a;
        }

        HD constexpr inline float3& operator-=(float3& a, const float3& b)
        {
            a.x -= b.x;
            a.y -= b.y;
            a.z -= b.z;
            return a;
        }

        HD constexpr inline float3& operator*=(float3& a, const float3& b)
        {
            a.x *= b.x;
            a.y *= b.y;
            a.z *= b.z;
            return a;
        }

        HD constexpr inline float3& operator/=(float3& a, const float3& b)
        {
            a.x /= b.x;
            a.y /= b.y;
            a.z /= b.z;
            return a;
        }

        HD constexpr inline float3& operator*=(float3& a, float b)
        {
            a.x *= b;
            a.y *= b;
            a.z *= b;
            return a;
        }

        HD constexpr inline float3& operator/=(float3& a, float b)
        {
            a.x /= b;
            a.y /= b;
            a.z /= b;
            return a;
        }

        // #################################################################### float4 operators ####################################################################

        HD constexpr inline float4 make_float4(float s)
        {
            return ::make_float4(s, s, s, s);
        }

        HD constexpr inline float4 operator+(const float4& a, const float4& b)
        {
            return ::make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
        }

        HD constexpr inline float4 operator-(const float4& a, const float4& b)
        {
            return ::make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
        }

        HD constexpr inline float4 operator*(const float4& a, const float4& b)
        {
            return ::make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
        }

        HD constexpr inline float4 operator/(const float4& a, const float4& b)
        {
            return ::make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
        }

        HD constexpr inline float4 operator*(const float4& a, float b)
        {
            return ::make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
        }

        HD constexpr inline float4 operator*(float a, const float4& b)
        {
            return ::make_float4(a * b.x, a * b.y, a * b.z, a * b.w);
        }

        HD constexpr inline float4 operator/(const float4& a, float b)
        {
            return ::make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
        }

        HD constexpr inline float4& operator+=(float4& a, const float4& b)
        {
            a.x += b.x;
            a.y += b.y;
            a.z += b.z;
            a.w += b.w;
            return a;
        }

        HD constexpr inline float4& operator-=(float4& a, const float4& b)
        {
            a.x -= b.x;
            a.y -= b.y;
            a.z -= b.z;
            a.w -= b.w;
            return a;
        }

        HD constexpr inline float4& operator*=(float4& a, const float4& b)
        {
            a.x *= b.x;
            a.y *= b.y;
            a.z *= b.z;
            a.w *= b.w;
            return a;
        }

        HD constexpr inline float4& operator/=(float4& a, const float4& b)
        {
            a.x /= b.x;
            a.y /= b.y;
            a.z /= b.z;
            a.w /= b.w;
            return a;
        }

        HD constexpr inline float4& operator*=(float4& a, float b)
        {
            a.x *= b;
            a.y *= b;
            a.z *= b;
            a.w *= b;
            return a;
        }

        HD constexpr inline float4& operator/=(float4& a, float b)
        {
            a.x /= b;
            a.y /= b;
            a.z /= b;
            a.w /= b;
            return a;
        }

        // other functions

        HD constexpr inline float length(const float2& v)
        {
            return ::sqrt(v.x * v.x + v.y * v.y);
        }

        HD constexpr inline float length(const float3& v)
        {
            return ::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        }

        HD constexpr inline float length(const float4& v)
        {
            return ::sqrt(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
        }

        HD constexpr inline float2 normalize(const float2& v)
        {
            return v / length(v);
        }

        HD constexpr inline float3 normalize(const float3& v)
        {
            return v / length(v);
        }

        HD constexpr inline float4 normalize(const float4& v)
        {
            return v / length(v);
        }

        HD constexpr inline float dot(const float2& a, const float2& b)
        {
            return a.x * b.x + a.y * b.y;
        }

        HD constexpr inline float dot(const float3& a, const float3& b)
        {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }

        HD constexpr inline float dot(const float4& a, const float4& b)
        {
            return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }

        HD constexpr inline float clamp(float value, float min, float max)
        {
            return value < min ? min : (value > max ? max : value);
        }

        HD constexpr inline float2 clamp(const float2& value, float2 min, float2 max)
        {
            return ::make_float2(
                clamp(value.x, min.x, max.x),
                clamp(value.y, min.y, max.y));
        }

        HD constexpr inline float3 clamp(const float3& value, float3 min, float3 max)
        {
            return ::make_float3(
                clamp(value.x, min.x, max.x),
                clamp(value.y, min.y, max.y),
                clamp(value.z, min.z, max.z));
        }

        HD constexpr inline float4 clamp(const float4& value, float4 min, float4 max)
        {
            return ::make_float4(
                clamp(value.x, min.x, max.x),
                clamp(value.y, min.y, max.y),
                clamp(value.z, min.z, max.z),
                clamp(value.w, min.w, max.w));
        }

        template <typename T>
        HD constexpr inline T saturate(T value)
        {
            return clamp(value, T(0), T(1));
        }

        HD constexpr inline float step(float edge, float x)
        {
            return x < edge ? 0.0f : 1.0f;
        }

        HD constexpr inline float smoothstep(float edge0, float edge1, float x)
        {
            const float t = saturate((x - edge0) / (edge1 - edge0));
            return t * t * (3.0f - 2.0f * t);
        }

        HD constexpr inline float2 smoothstep(const float2& edge0, const float2& edge1, const float2& x)
        {
            return ::make_float2(
                smoothstep(edge0.x, edge1.x, x.x),
                smoothstep(edge0.y, edge1.y, x.y));
        }

        HD constexpr inline float3 smoothstep(const float3& edge0, const float3& edge1, const float3& x)
        {
            return ::make_float3(
                smoothstep(edge0.x, edge1.x, x.x),
                smoothstep(edge0.y, edge1.y, x.y),
                smoothstep(edge0.z, edge1.z, x.z));
        }

        HD constexpr inline float4 smoothstep(const float4& edge0, const float4& edge1, const float4& x)
        {
            return ::make_float4(
                smoothstep(edge0.x, edge1.x, x.x),
                smoothstep(edge0.y, edge1.y, x.y),
                smoothstep(edge0.z, edge1.z, x.z),
                smoothstep(edge0.w, edge1.w, x.w));
        }

        HD constexpr inline float smootherstep(float edge0, float edge1, float x)
        {
            const float t = saturate((x - edge0) / (edge1 - edge0));
            return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
        }

        HD constexpr inline float2 smootherstep(const float2& edge0, const float2& edge1, const float2& x)
        {
            return ::make_float2(
                smootherstep(edge0.x, edge1.x, x.x),
                smootherstep(edge0.y, edge1.y, x.y));
        }

        HD constexpr inline float3 smootherstep(const float3& edge0, const float3& edge1, const float3& x)
        {
            return ::make_float3(
                smootherstep(edge0.x, edge1.x, x.x),
                smootherstep(edge0.y, edge1.y, x.y),
                smootherstep(edge0.z, edge1.z, x.z));
        }

        HD constexpr inline float4 smootherstep(const float4& edge0, const float4& edge1, const float4& x)
        {
            return ::make_float4(
                smootherstep(edge0.x, edge1.x, x.x),
                smootherstep(edge0.y, edge1.y, x.y),
                smootherstep(edge0.z, edge1.z, x.z),
                smootherstep(edge0.w, edge1.w, x.w));
        }

        template <typename T>
        HD constexpr inline T lerp(T a, T b, float t)
        {
            return a + t * (b - a);
        }

        HD constexpr inline float lerp(float a, float b, float t)
        {
            return ::fma(t, b, ::fma(-t, a, a));
        }

        HD constexpr inline float2 lerp(const float2& a, const float2& b, float t)
        {
            return ::make_float2(
                lerp(a.x, b.x, t),
                lerp(a.y, b.y, t));
        }

        HD constexpr inline float3 lerp(const float3& a, const float3& b, float t)
        {
            return ::make_float3(
                lerp(a.x, b.x, t),
                lerp(a.y, b.y, t),
                lerp(a.z, b.z, t));
        }

        HD constexpr inline float4 lerp(const float4& a, const float4& b, float t)
        {
            return ::make_float4(
                lerp(a.x, b.x, t),
                lerp(a.y, b.y, t),
                lerp(a.z, b.z, t),
                lerp(a.w, b.w, t));
        }

        HD constexpr inline float2 fma(const float2& a, const float2& b, const float2& c)
        {
            return ::make_float2(::fma(a.x, b.x, c.x), ::fma(a.y, b.y, c.y));
        }

        HD constexpr inline float3 fma(const float3& a, const float3& b, const float3& c)
        {
            return ::make_float3(::fma(a.x, b.x, c.x), ::fma(a.y, b.y, c.y), ::fma(a.z, b.z, c.z));
        }

        HD constexpr inline float4 fma(const float4& a, const float4& b, const float4& c)
        {
            return ::make_float4(::fma(a.x, b.x, c.x), ::fma(a.y, b.y, c.y), ::fma(a.z, b.z, c.z), ::fma(a.w, b.w, c.w));
        }

    } // namespace utils
} // namespace roesti
