#include "rasterizer.cuh"

using namespace roesti;
using namespace roesti::utils;

constexpr float epsilon = 1e-6f;

KERNEL void clearBuffers(float4* __restrict__ depthBufferColor, int* __restrict__ depthBufferLayer, int width, int height, const float4* __restrict__ clearColor, int clearLayer)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        depthBufferColor[y * width + x] = clearColor[0];
        depthBufferLayer[y * width + x] = static_cast<int>(round(clearLayer));
    }
}

KERNEL void vertexShaderKernel(const float2* __restrict__ inVerticesPositions, const float4* __restrict__ inVerticesColors, const float2* __restrict__ inVerticesTexcoords,
                               const int* __restrict__ inVerticesLayers, float2* __restrict__ outVerticesPositions, float4* __restrict__ outVerticesColors,
                               float2* __restrict__ outVerticesTexcoords, int* __restrict__ outVerticesLayers, int vertexCount,
                               int width, int height)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < vertexCount)
    {
        // Compute aspect ratio
        float aspect = float(width) / float(height);

        // Simple orthographic projection with aspect correction
        float2 ndc = inVerticesPositions[idx];

        ndc.x *= aspect > 1.0f ? 1.0f / aspect : 1.0f;
        ndc.y *= aspect < 1.0f ? aspect : 1.0f;

        // Vulkan NDC Y is flipped compared to OpenGL
        ndc.y = -ndc.y;

        // Center alignment: map NDC [-1,1] to pixel centers
        outVerticesPositions[idx].x = ((ndc.x + 1.0f) * 0.5f) * float(width) - 0.5f;
        outVerticesPositions[idx].y = ((ndc.y + 1.0f) * 0.5f) * float(height) - 0.5f;

        outVerticesColors[idx]    = inVerticesColors[idx];
        outVerticesTexcoords[idx] = inVerticesTexcoords[idx];
        outVerticesLayers[idx]    = inVerticesLayers[idx];
    }
}

DEVICE FORCE_INLINE float orient2D(float2 a, float2 b, float2 c)
{
    return fma((b.x - a.x), (c.y - a.y), -((b.y - a.y) * (c.x - a.x)));
}

KERNEL void assembleTrianglesKernel(const float2* __restrict__ vertexPositions, const float4* __restrict__ vertexColors, const float2* __restrict__ vertexTexcoords,
                                    const int* __restrict__ vertexLayers, const int* __restrict__ indices,
                                    float2* __restrict__ primitivePositions, float4* __restrict__ primitiveColors, float2* __restrict__ primitiveTexcoords, int* __restrict__ primitiveLayers,
                                    float4* __restrict__ primitiveAABBs, int primitiveCount, PrimitiveType primitiveType,
                                    int width, int height, int lineWidth)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < primitiveCount)
    {
        int vertexStart  = (primitiveType == PrimitiveType::Triangles) ? idx * 3 : idx;
        float2& primPos0 = primitivePositions[idx * 3]; // since we always store 3 vertices per primitive
        float2& primPos1 = primitivePositions[idx * 3 + 1];
        float2& primPos2 = primitivePositions[idx * 3 + 2];

        float4& primCol0 = primitiveColors[idx * 3];
        float4& primCol1 = primitiveColors[idx * 3 + 1];
        float4& primCol2 = primitiveColors[idx * 3 + 2];

        float2& primTex0 = primitiveTexcoords[idx * 3];
        float2& primTex1 = primitiveTexcoords[idx * 3 + 1];
        float2& primTex2 = primitiveTexcoords[idx * 3 + 2];

        int& primLayer0 = primitiveLayers[idx * 3];
        int& primLayer1 = primitiveLayers[idx * 3 + 1];
        int& primLayer2 = primitiveLayers[idx * 3 + 2];

        const int& i0 = indices[vertexStart];
        const int& i1 = indices[vertexStart + 1];
        const int& i2 = indices[vertexStart + 2];

        bool isDegenerate = false;

        if (primitiveType == PrimitiveType::Triangles)
        {
            primPos0 = vertexPositions[i0];
            primPos1 = vertexPositions[i1];
            primPos2 = vertexPositions[i2];

            primCol0 = vertexColors[i0];
            primCol1 = vertexColors[i1];
            primCol2 = vertexColors[i2];

            primTex0 = vertexTexcoords[i0];
            primTex1 = vertexTexcoords[i1];
            primTex2 = vertexTexcoords[i2];

            primLayer0 = vertexLayers[i0];
            primLayer1 = vertexLayers[i1];
            primLayer2 = vertexLayers[i2];
        }
        else // TriangleStrip
        {
            // Even triangles: CCW, Odd triangles: CW (swap first two vertices)
            if (idx % 2 == 0)
            {
                primPos0 = vertexPositions[i0];
                primPos1 = vertexPositions[i1];
                primPos2 = vertexPositions[i2];

                primCol0 = vertexColors[i0];
                primCol1 = vertexColors[i1];
                primCol2 = vertexColors[i2];

                primTex0 = vertexTexcoords[i0];
                primTex1 = vertexTexcoords[i1];
                primTex2 = vertexTexcoords[i2];

                primLayer0 = vertexLayers[i0];
                primLayer1 = vertexLayers[i1];
                primLayer2 = vertexLayers[i2];
            }
            else
            {
                primPos0 = vertexPositions[i1];
                primPos1 = vertexPositions[i0];
                primPos2 = vertexPositions[i2];

                primCol0 = vertexColors[i1];
                primCol1 = vertexColors[i0];
                primCol2 = vertexColors[i2];

                primTex0 = vertexTexcoords[i1];
                primTex1 = vertexTexcoords[i0];
                primTex2 = vertexTexcoords[i2];

                primLayer0 = vertexLayers[i1];
                primLayer1 = vertexLayers[i0];
                primLayer2 = vertexLayers[i2];
            }
        }

        // Cull backfacing triangles (CCW winding is front-facing)
        // Degenerate check
        {
            const float2& p0 = primPos0;
            const float2& p1 = primPos1;
            const float2& p2 = primPos2;
            float area       = orient2D(p0, p1, p2);
            if (area <= epsilon)
            {
                isDegenerate = true;
            }
            if ((abs(p0.x - p1.x) < epsilon && abs(p0.y - p1.y) < epsilon) ||
                (abs(p0.x - p2.x) < epsilon && abs(p0.y - p2.y) < epsilon) ||
                (abs(p1.x - p2.x) < epsilon && abs(p1.y - p2.y) < epsilon))
            {
                isDegenerate = true;
            }
        }

        float minX, minY, maxX, maxY;
        if (primitiveType == PrimitiveType::TriangleStrip && isDegenerate)
        {
            minX = minY = maxX = maxY = -1.0f;
        }
        else
        {
            minX = min(primPos0.x, min(primPos1.x, primPos2.x));
            minY = min(primPos0.y, min(primPos1.y, primPos2.y));
            maxX = max(primPos0.x, max(primPos1.x, primPos2.x));
            maxY = max(primPos0.y, max(primPos1.y, primPos2.y));
        }

        float4& minMax = primitiveAABBs[idx];

        minMax = make_float4(
            clamp(minX, 0.0f, float(width - 1)),
            clamp(minY, 0.0f, float(height - 1)),
            clamp(maxX, 0.0f, float(width - 1)),
            clamp(maxY, 0.0f, float(height - 1)));

        // if min == max set to -1,-1 to indicate degenerate
        if (minMax.x >= minMax.z || minMax.y >= minMax.w)
        {
            minMax = make_float4(-1.0f);
        }
    }
}

KERNEL void assembleLinesKernel(const float2* __restrict__ vertexPositions, const float4* __restrict__ vertexColors, const float2* __restrict__ vertexTexcoords,
                                const int* __restrict__ vertexLayers, const int* __restrict__ indices,
                                float2* __restrict__ primitivePositions, float4* __restrict__ primitiveColors, float2* __restrict__ primitiveTexcoords, int* __restrict__ primitiveLayers,
                                float4* __restrict__ primitiveAABBs, int primitiveCount, PrimitiveType primitiveType,
                                int width, int height, int lineWidth)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < primitiveCount)
    {
        int vertexStart  = (primitiveType == PrimitiveType::Lines) ? idx * 2 : idx;
        float2& primPos0 = primitivePositions[idx * 3]; // since we always store 3 vertices per primitive
        float2& primPos1 = primitivePositions[idx * 3 + 1];

        primPos0 = vertexPositions[indices[vertexStart]];
        primPos1 = vertexPositions[indices[vertexStart + 1]];

        float4& primCol0 = primitiveColors[idx * 3];
        float4& primCol1 = primitiveColors[idx * 3 + 1];

        primCol0 = vertexColors[indices[vertexStart]];
        primCol1 = vertexColors[indices[vertexStart + 1]];

        float2& primTex0 = primitiveTexcoords[idx * 3];
        float2& primTex1 = primitiveTexcoords[idx * 3 + 1];

        primTex0 = vertexTexcoords[indices[vertexStart]];
        primTex1 = vertexTexcoords[indices[vertexStart + 1]];

        int& primLayer0 = primitiveLayers[idx * 3];
        int& primLayer1 = primitiveLayers[idx * 3 + 1];

        primLayer0 = vertexLayers[indices[vertexStart]];
        primLayer1 = vertexLayers[indices[vertexStart + 1]];

        float minX = min(primPos0.x, primPos1.x);
        float minY = min(primPos0.y, primPos1.y);
        float maxX = max(primPos0.x, primPos1.x);
        float maxY = max(primPos0.y, primPos1.y);

        if (lineWidth > 1)
        {
            float halfWidth = ceilf(0.5f * lineWidth);
            minX -= halfWidth;
            minY -= halfWidth;
            maxX += halfWidth;
            maxY += halfWidth;
        }

        float4& minMax = primitiveAABBs[idx];

        minMax = make_float4(max(minX, 0.0f), max(minY, 0.0f), min(maxX, float(width - 1)), min(maxY, float(height - 1)));

        // if min == max set to -1,-1 to indicate degenerate
        if (minMax.x >= minMax.z || minMax.y >= minMax.w)
        {
            minMax = make_float4(-1.0f, -1.0f, -1.0f, -1.0f);
        }
    }
}

KERNEL void assemblePointsKernel(const float2* __restrict__ vertexPositions, const float4* __restrict__ vertexColors, const float2* __restrict__ vertexTexcoords,
                                 const int* __restrict__ vertexLayers, const int* __restrict__ indices,
                                 float2* __restrict__ primitivePositions, float4* __restrict__ primitiveColors, float2* __restrict__ primitiveTexcoords, int* __restrict__ primitiveLayers,
                                 float4* __restrict__ primitiveAABBs, int primitiveCount, PrimitiveType primitiveType,
                                 int width, int height, int pointSize)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < primitiveCount)
    {
        float2& primPos = primitivePositions[idx * 3]; // since we always store 3 vertices per primitive

        primPos = vertexPositions[indices[idx]];

        float4& primCol = primitiveColors[idx * 3];

        primCol = vertexColors[indices[idx]];

        float2& primTex = primitiveTexcoords[idx * 3];

        primTex = vertexTexcoords[indices[idx]];

        int& primLayer = primitiveLayers[idx * 3];

        primLayer = vertexLayers[indices[idx]];

        float radius = pointSize > 1 ? 0.5f * pointSize : 0.5f;
        float minX   = primPos.x - radius;
        float minY   = primPos.y - radius;
        float maxX   = primPos.x + radius;
        float maxY   = primPos.y + radius;

        float4& minMax = primitiveAABBs[idx];

        minMax = make_float4(
            clamp(minX, 0.0f, float(width - 1)),
            clamp(minY, 0.0f, float(height - 1)),
            clamp(maxX, 0.0f, float(width - 1)),
            clamp(maxY, 0.0f, float(height - 1)));

        // if min == max set to -1,-1 to indicate degenerate
        if (minMax.x >= minMax.z || minMax.y >= minMax.w)
        {
            minMax = make_float4(-1.0f);
        }
    }
}

template <typename FragmentShader>
DEVICE FORCE_INLINE bool drawFragment(float4* __restrict__ depthBufferColor, int* __restrict__ depthBufferLayer, int index, float2 fragPos, float2 texCoord, int primitiveId, BlendMode blendMode, FragmentShader fragmentShaderFunc, int layer, float4 color)
{
    Fragment frag;
    if (blendMode == BlendMode::Opaque)
    {
        atomicMax(&depthBufferLayer[index], layer);
        if (layer == depthBufferLayer[index])
        {
            frag.color           = color;
            frag.layer           = layer;
            const bool discarded = fragmentShaderFunc(frag, fragPos, texCoord, primitiveId);

            if (discarded)
            {
                // reset depth layer if fragment is discarded
                atomicMax(&depthBufferLayer[index], layer - 1);
                return false;
            }

            depthBufferColor[index] = frag.color;

            return true;
        }
    }
    else if (blendMode == BlendMode::AlphaBlendF2B)
    {
        frag.color = color;
        // frag.layer     = layer;
        const bool discarded = fragmentShaderFunc(frag, fragPos, texCoord, primitiveId);

        if (discarded)
        {
            // reset depth layer if fragment is discarded
            return false;
        }
        // Front-to-back alpha blending
        float4& dstColor       = depthBufferColor[index];
        const float4& srcColor = frag.color;

        const float outAlpha = (1.0f - dstColor.w) * srcColor.w;

        dstColor.x = fma(srcColor.x, outAlpha, dstColor.x);
        dstColor.y = fma(srcColor.y, outAlpha, dstColor.y);
        dstColor.z = fma(srcColor.z, outAlpha, dstColor.z);
        dstColor.w = dstColor.w + outAlpha;

        depthBufferColor[index] = dstColor;

        return dstColor.w > 0.99999f; // consider fully opaque
    }
}

DEVICE FORCE_INLINE float3 calculateBarycentric(float2 a, float2 b, float2 c, float px, float py)
{
    const float area = orient2D(a, b, c);
    if (abs(area) < 1e-8f)
    {
        return make_float3(-1.0f); // Degenerate triangle
    }

    const float invArea = __frcp_rn(area);

    const float w0 = orient2D(b, c, make_float2(px, py)) * invArea;
    const float w1 = orient2D(c, a, make_float2(px, py)) * invArea;
    const float w2 = orient2D(a, b, make_float2(px, py)) * invArea;

    return make_float3(w0, w1, w2);
}

struct BarycentricAndDerivatives
{
    float3 bary;
    float3 ddx;
    float3 ddy;
};

// similar to http://filmicworlds.com/blog/visibility-buffer-rendering-with-material-graphs/
DEVICE FORCE_INLINE BarycentricAndDerivatives calculateBarycentricAndDerivatives(float2 a, float2 b, float2 c, float px, float py)
{
    float area = orient2D(a, b, c);
    if (abs(area) < 1e-8f)
    {
        BarycentricAndDerivatives result;
        result.bary = make_float3(-1.0f);
        result.ddx  = make_float3(0.0f);
        result.ddy  = make_float3(0.0f);
        return result;
    }

    float invArea = __frcp_rn(area);

    float w0    = orient2D(b, c, make_float2(px, py)) * invArea;
    float w1    = orient2D(c, a, make_float2(px, py)) * invArea;
    float w2    = orient2D(a, b, make_float2(px, py)) * invArea;
    float3 bary = make_float3(w0, w1, w2);

    float3 bary_dx, bary_dy;

    {
        float w0_dx = orient2D(b, c, make_float2(px + 1.0f, py)) * invArea;
        float w1_dx = orient2D(c, a, make_float2(px + 1.0f, py)) * invArea;
        float w2_dx = orient2D(a, b, make_float2(px + 1.0f, py)) * invArea;
        bary_dx     = make_float3(w0_dx, w1_dx, w2_dx) - bary;

        float w0_dy = orient2D(b, c, make_float2(px, py + 1.0f)) * invArea;
        float w1_dy = orient2D(c, a, make_float2(px, py + 1.0f)) * invArea;
        float w2_dy = orient2D(a, b, make_float2(px, py + 1.0f)) * invArea;
        bary_dy     = make_float3(w0_dy, w1_dy, w2_dy) - bary;
    }

    BarycentricAndDerivatives result;
    result.bary = bary;
    result.ddx  = bary_dx;
    result.ddy  = bary_dy;
    return result;
}

DEVICE FORCE_INLINE bool barycentricInBounds(const float3& bary)
{
    // Exclude if bary.x is left edge or bary.y is top edge
    return bary.x > 0.0f && bary.y > 0.0f && bary.z >= 0.0f;
}

DEVICE FORCE_INLINE bool onBorder(float2 a, float2 b, float2 c, float px, float py, int lineWidth = 1)
{
    if (lineWidth <= 1)
    {
        lineWidth = 1;
    }

    BarycentricAndDerivatives baryAndDerivs = calculateBarycentricAndDerivatives(a, b, c, px, py);

    if (!barycentricInBounds(baryAndDerivs.bary))
    {
        return false;
    }

    float3 bary       = baryAndDerivs.bary;
    const float3& ddx = baryAndDerivs.ddx;
    const float3& ddy = baryAndDerivs.ddy;

    // Use derivatives to compute wireframe thickness in screen space
    // similar to https://catlikecoding.com/unity/tutorials/advanced-rendering/flat-and-wireframe-shading/

    const float3 deltas = make_float3(
        abs(ddx.x) + abs(ddy.x),
        abs(ddx.y) + abs(ddy.y),
        abs(ddx.z) + abs(ddy.z));

    const float3 thickness = deltas * float(lineWidth);

    const float3 barys = smoothstep(thickness, thickness * 2.0f, bary);

    const float minBary = min(barys.x, min(barys.y, barys.z));

    return minBary < 0.5f;
}

KERNEL void computeTilesTouchedKernel(const float4* __restrict__ aabbs, const int3* __restrict__ layers, int primitiveCount, PrimitiveType primitiveType, int* __restrict__ tilesTouched, int* __restrict__ depths, int width, int height)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < primitiveCount)
    {
        tilesTouched[idx] = 0;

        const float4& minMax = aabbs[idx];

        if (minMax.x < 0.0f || minMax.y < 0.0f || minMax.z < 0.0f || minMax.w < 0.0f)
        {
            // Degenerate or culled primitive, touches no tiles
            tilesTouched[idx] = 0;
        }
        else
        {
            // inclusive min, exclusive max
            const int tileMinX = max(0, int(floor(minMax.x / TILE_WIDTH)));
            const int tileMinY = max(0, int(floor(minMax.y / TILE_HEIGHT)));
            const int tileMaxX = min((width + TILE_WIDTH - 1) / TILE_WIDTH, int(ceil(minMax.z / TILE_WIDTH) + 1));
            const int tileMaxY = min((height + TILE_HEIGHT - 1) / TILE_HEIGHT, int(ceil(minMax.w / TILE_HEIGHT) + 1));

            const int numTilesX = max(0, tileMaxX - tileMinX);
            const int numTilesY = max(0, tileMaxY - tileMinY);

            tilesTouched[idx] = numTilesX * numTilesY;
        }

        const int3& layer = layers[idx];

        // depth is average layer but since we do not know here which type of primitive it is, we just store the layer of v0
        depths[idx] = primitiveType == PrimitiveType::Points ? layer.x : (primitiveType == PrimitiveType::Lines || primitiveType == PrimitiveType::LineStrip) ? (layer.x + layer.y) / 2.0f
                                                                                                                                                              : (layer.x + layer.y + layer.z) / 3.0f;
    }
}

KERNEL void createTileLookupKernel(const float4* __restrict__ aabbs, int primitiveCount, const int* __restrict__ tilesTouchedScan, int* __restrict__ depths, int2 tileGrid, int width, int height, int64_t* __restrict__ keys, int* __restrict__ values)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < primitiveCount)
    {
        const float4& minMax = aabbs[idx];

        if (minMax.x < 0.0f || minMax.y < 0.0f || minMax.z < 0.0f || minMax.w < 0.0f)
        {
            // Degenerate or culled primitive
            return;
        }

        // inclusive min, exclusive max
        const int tileMinX = max(0, int(floor(minMax.x / TILE_WIDTH)));
        const int tileMinY = max(0, int(floor(minMax.y / TILE_HEIGHT)));
        const int tileMaxX = min((width + TILE_WIDTH - 1) / TILE_WIDTH, int(ceil(minMax.z / TILE_WIDTH) + 1));
        const int tileMaxY = min((height + TILE_HEIGHT - 1) / TILE_HEIGHT, int(ceil(minMax.w / TILE_HEIGHT) + 1));

        // Similar to GS
        int currentIndex = idx == 0 ? 0 : tilesTouchedScan[idx - 1];

        const int32_t depthIdx = *((int32_t*)&(depths[idx]));
        for (int ty = tileMinY; ty < tileMaxY; ++ty)
        {
            for (int tx = tileMinX; tx < tileMaxX; ++tx)
            {
                int64_t key = static_cast<int64_t>(ty * tileGrid.x + tx);
                key <<= 32;                             // Shift key to the left to make space for depth index
                key |= static_cast<uint32_t>(depthIdx); // Combine tile index and depth index
                keys[currentIndex]   = key;
                values[currentIndex] = idx;
                currentIndex++;
            }
        }
    }
}

KERNEL void getTileRangesKernel(const int64_t* __restrict__ sortedKeys, int totalKeys, int2* __restrict__ tileRanges)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < totalKeys)
    {
        const int currentTileIdx = int(sortedKeys[idx] >> 32);

        if (idx == 0)
        {
            tileRanges[currentTileIdx].x = idx; // Start of the first tile range
        }
        else
        {
            // previous tile
            const int prevTileIdx = int(sortedKeys[idx - 1] >> 32);

            // if the tile index changes, we have a new tile range
            if (currentTileIdx != prevTileIdx)
            {
                tileRanges[prevTileIdx].y    = idx; // End of the previous tile range
                tileRanges[currentTileIdx].x = idx; // Start of the new tile range
            }
        }

        // Handle the last element
        if (idx == totalKeys - 1)
        {
            tileRanges[currentTileIdx].y = totalKeys; // End of the last tile range
        }
    }
}

// One thread per pixel in the tile, one thread block per tile.
// Collaboratively works on on per block, each thread treats one pixel in the tile.
template <typename FragmentShader>
KERNEL void __launch_bounds__(TILE_SIZE) rasterizeTrianglesTiledKernel(
    const float2* __restrict__ primitivePositions,
    const float4* __restrict__ primitiveColors,
    const float2* __restrict__ primitiveTexcoords,
    const int* __restrict__ primitiveLayers,
    int primitiveCount,
    float4* __restrict__ depthBufferColor,
    int* __restrict__ depthBufferLayer,
    BlendMode blendMode,
    FragmentShader fragmentShaderFunc,
    int lineWidth,
    bool wireframe,
    int width, int height,
    int2 tileGrid, int2* __restrict__ tileRanges, int* __restrict__ lookupValues)
{
    const int tileIndex = blockIdx.y * tileGrid.x + blockIdx.x;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    const int pixelIdx = j * width + i;

    const float px = float(i);
    const float py = float(j);

    // Check if the thread is within the image bounds. If not, this thread will only fetch data and not process it.
    const bool insideImageBounds = (i < width && j < height);

    bool done = !insideImageBounds; // thread will still fetch data even if outside image bounds

    const int2& tileRange = tileRanges[tileIndex];
    const int batchCount  = (tileRange.y - tileRange.x + TILE_SIZE - 1) / TILE_SIZE; // number of batches to process all primitives in the tile

    struct SharedPrimitiveData
    {
        float4 c0, c1, c2;
        float2 p0, p1, p2;
        float2 tex0, tex1, tex2;
        int l0, l1, l2, id;
        int2 padding; // Padding to make the size a multiple of 32 bytes
    };
    __shared__ SharedPrimitiveData sharedPrimitives[TILE_SIZE];

    const uint threadRank = threadIdx.y * blockDim.x + threadIdx.x;

    // Iterate over batches of primitives
    for (int batch = 0; batch < batchCount; ++batch)
    {
        // syncthreads to ensure all threads are ready before beginning the next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= TILE_SIZE)
        {
            break;
        }

        // Load a batch of primitives into shared memory
        const int batchStart = tileRange.x + batch * TILE_SIZE;
        const int primIdx    = batchStart + threadRank;
        if (primIdx < tileRange.y)
        {
            const int primitiveId = lookupValues[primIdx];

            SharedPrimitiveData& sp = sharedPrimitives[threadRank];

            sp.id   = primitiveId;
            sp.p0   = primitivePositions[primitiveId * 3 + 0];
            sp.p1   = primitivePositions[primitiveId * 3 + 1];
            sp.p2   = primitivePositions[primitiveId * 3 + 2];
            sp.tex0 = primitiveTexcoords[primitiveId * 3 + 0];
            sp.tex1 = primitiveTexcoords[primitiveId * 3 + 1];
            sp.tex2 = primitiveTexcoords[primitiveId * 3 + 2];
            sp.c0   = primitiveColors[primitiveId * 3 + 0];
            sp.c1   = primitiveColors[primitiveId * 3 + 1];
            sp.c2   = primitiveColors[primitiveId * 3 + 2];
            sp.l0   = primitiveLayers[primitiveId * 3 + 0];
            sp.l1   = primitiveLayers[primitiveId * 3 + 1];
            sp.l2   = primitiveLayers[primitiveId * 3 + 2];
        }

        __barrier_sync(0); // Ensure all threads have loaded their primitive data

        const int batchSize = min(TILE_SIZE, tileRange.y - batchStart);

        // Process the loaded primitives
        for (int k = 0; k < batchSize && !done; ++k)
        {
            // Access the primitive data from shared memory
            const SharedPrimitiveData& sp = sharedPrimitives[k];

            const float3 bary = calculateBarycentric(sp.p0, sp.p1, sp.p2, px, py); // FIXME: wireframe computes bary twice
            if ((!wireframe && barycentricInBounds(bary)) || onBorder(sp.p0, sp.p1, sp.p2, px, py, lineWidth))
            {
                const float& w0 = bary.x;
                const float& w1 = bary.y;
                const float& w2 = bary.z;

                float4 color;
                color.x = fma(w0, sp.c0.x, fma(w1, sp.c1.x, w2 * sp.c2.x));
                color.y = fma(w0, sp.c0.y, fma(w1, sp.c1.y, w2 * sp.c2.y));
                color.z = fma(w0, sp.c0.z, fma(w1, sp.c1.z, w2 * sp.c2.z));
                color.w = fma(w0, sp.c0.w, fma(w1, sp.c1.w, w2 * sp.c2.w));

                float2 texCoord;
                texCoord.x = fma(w0, sp.tex0.x, fma(w1, sp.tex1.x, w2 * sp.tex2.x));
                texCoord.y = fma(w0, sp.tex0.y, fma(w1, sp.tex1.y, w2 * sp.tex2.y));

                int layer = static_cast<int>(fma(w0, float(sp.l0), fma(w1, float(sp.l1), w2 * float(sp.l2))));

                if (drawFragment(depthBufferColor, depthBufferLayer, pixelIdx, make_float2(px, py), texCoord, sp.id, blendMode, fragmentShaderFunc, layer, color))
                {
                    done = true; // stop processing more primitives for this pixel if the fragment was drawn fully opaque
                    break;
                }
            }
        }
    }
}

template <typename FragmentShader>
KERNEL void __launch_bounds__(TILE_SIZE) rasterizePointsTiledKernel(
    const float2* __restrict__ primitivePositions,
    const float4* __restrict__ primitiveColors,
    const float2* __restrict__ primitiveTexcoords,
    const int* __restrict__ primitiveLayers,
    int primitiveCount,
    float4* __restrict__ depthBufferColor,
    int* __restrict__ depthBufferLayer,
    BlendMode blendMode,
    FragmentShader fragmentShaderFunc,
    int pointSize,
    int width, int height,
    int2 tileGrid, int2* __restrict__ tileRanges, int* __restrict__ lookupValues)
{
    const int tileIndex = blockIdx.y * tileGrid.x + blockIdx.x;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    const int pixelIdx = j * width + i;

    const float px = float(i);
    const float py = float(j);

    // Check if the thread is within the image bounds. If not, this thread will only fetch data and not process it.
    const bool insideImageBounds = (i < width && j < height);

    bool done = !insideImageBounds;

    const int2& tileRange = tileRanges[tileIndex];
    const int batchCount  = (tileRange.y - tileRange.x + TILE_SIZE - 1) / TILE_SIZE;

    struct SharedPrimitiveData
    {
        float4 c0;
        float2 p0;
        float2 tex0;
        int l0;
        int id;
        int2 padding; // Padding to make the size a multiple of 32 bytes
    };
    __shared__ SharedPrimitiveData sharedPrimitives[TILE_SIZE];

    const uint threadRank = threadIdx.y * blockDim.x + threadIdx.x;

    for (int batch = 0; batch < batchCount; ++batch)
    {
        if (__syncthreads_count(done) >= TILE_SIZE)
            break;

        const int batchStart = tileRange.x + batch * TILE_SIZE;
        const int primIdx    = batchStart + threadRank;
        if (primIdx < tileRange.y)
        {
            const int primitiveId = lookupValues[primIdx];

            SharedPrimitiveData& sp = sharedPrimitives[threadRank];

            sp.id   = primitiveId;
            sp.p0   = primitivePositions[primitiveId * 3 + 0];
            sp.c0   = primitiveColors[primitiveId * 3 + 0];
            sp.tex0 = primitiveTexcoords[primitiveId * 3 + 0];
            sp.l0   = primitiveLayers[primitiveId * 3 + 0];
        }

        __barrier_sync(0);

        const int batchSize = min(TILE_SIZE, tileRange.y - batchStart);

        for (int k = 0; k < batchSize && !done; ++k)
        {
            const SharedPrimitiveData& sp = sharedPrimitives[k];

            const float radius = pointSize > 1 ? 0.5f * pointSize : 0.5f;
            const float dx     = px - sp.p0.x;
            const float dy     = py - sp.p0.y;
            if (dx * dx + dy * dy <= radius * radius)
            {
                if (drawFragment(depthBufferColor, depthBufferLayer, pixelIdx, make_float2(px, py), sp.tex0, sp.id, blendMode, fragmentShaderFunc, sp.l0, sp.c0))
                {
                    done = true; // stop processing more primitives for this pixel if the fragment was drawn fully opaque
                    break;
                }
            }
        }
    }
}

template <typename FragmentShader>
KERNEL void __launch_bounds__(TILE_SIZE) rasterizeLinesTiledKernel(
    const float2* __restrict__ primitivePositions,
    const float4* __restrict__ primitiveColors,
    const float2* __restrict__ primitiveTexcoords,
    const int* __restrict__ primitiveLayers,
    int primitiveCount,
    float4* __restrict__ depthBufferColor,
    int* __restrict__ depthBufferLayer,
    BlendMode blendMode,
    FragmentShader fragmentShaderFunc,
    int lineWidth,
    int width, int height,
    int2 tileGrid, int2* __restrict__ tileRanges, int* __restrict__ lookupValues)
{
    const int tileIndex = blockIdx.y * tileGrid.x + blockIdx.x;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    const int pixelIdx = j * width + i;

    const float px = float(i);
    const float py = float(j);

    // Check if the thread is within the image bounds. If not, this thread will only fetch data and not process it.
    const bool insideImageBounds = (i < width && j < height);

    bool done = !insideImageBounds;

    const int2& tileRange = tileRanges[tileIndex];
    const int batchCount  = (tileRange.y - tileRange.x + TILE_SIZE - 1) / TILE_SIZE;

    struct SharedPrimitiveData
    {
        float4 c0, c1;
        float2 p0, p1;
        float2 tex0, tex1;
        int l0, l1, id;
        int padding;
    };
    __shared__ SharedPrimitiveData sharedPrimitives[TILE_SIZE];

    uint threadRank = threadIdx.y * blockDim.x + threadIdx.x;

    for (int batch = 0; batch < batchCount; ++batch)
    {
        if (__syncthreads_count(done) >= TILE_SIZE)
            break;

        const int batchStart = tileRange.x + batch * TILE_SIZE;
        const int primIdx    = batchStart + threadRank;
        if (primIdx < tileRange.y)
        {
            const int primitiveId = lookupValues[primIdx];

            SharedPrimitiveData& sp = sharedPrimitives[threadRank];

            sp.id   = primitiveId;
            sp.p0   = primitivePositions[primitiveId * 3 + 0];
            sp.p1   = primitivePositions[primitiveId * 3 + 1];
            sp.tex0 = primitiveTexcoords[primitiveId * 3 + 0];
            sp.tex1 = primitiveTexcoords[primitiveId * 3 + 1];
            sp.c0   = primitiveColors[primitiveId * 3 + 0];
            sp.c1   = primitiveColors[primitiveId * 3 + 1];
            sp.l0   = primitiveLayers[primitiveId * 3 + 0];
            sp.l1   = primitiveLayers[primitiveId * 3 + 1];
        }

        __barrier_sync(0);

        const int batchSize = min(TILE_SIZE, tileRange.y - batchStart);

        for (int k = 0; k < batchSize && !done; ++k)
        {
            const SharedPrimitiveData& sp = sharedPrimitives[k];

            // Compute distance from pixel center to the line segment
            const float2 a = sp.p0;
            const float2 b = sp.p1;
            const float2 p = make_float2(px, py);

            const float2 ab      = b - a;
            const float2 ap      = p - a;
            const float abLenSq  = ab.x * ab.x + ab.y * ab.y;
            const float t        = abLenSq > 0 ? max(0.0f, min(1.0f, (ap.x * ab.x + ap.y * ab.y) / abLenSq)) : 0.0f;
            const float2 closest = make_float2(fma(ab.x, t, a.x), fma(ab.y, t, a.y));
            const float dist2    = (p.x - closest.x) * (p.x - closest.x) + (p.y - closest.y) * (p.y - closest.y);

            const float halfWidth = lineWidth > 1 ? 0.5f * lineWidth : 0.5f;

            if (dist2 <= halfWidth * halfWidth)
            {
                const float4 color    = lerp(sp.c0, sp.c1, t);
                const float2 texCoord = lerp(sp.tex0, sp.tex1, t);
                const int layer       = static_cast<int>(utils::lerp(float(sp.l0), float(sp.l1), t));

                if (drawFragment(depthBufferColor, depthBufferLayer, pixelIdx, make_float2(px, py), texCoord, sp.id, blendMode, fragmentShaderFunc, layer, color))
                {
                    done = true; // stop processing more primitives for this pixel if the fragment was drawn fully opaque
                    break;
                }
            }
        }
    }
}

#define discard return true
#define keep return false

DEVICE FORCE_INLINE bool fragmentShader(Fragment& fragment, float2 fragPos, float2 texCoord, int primitiveId)
{
    // do nothing

    keep;
}

struct FragmentShaderFunctor
{
    DEVICE bool operator()(Fragment& fragment, float2 fragPos, float2 texCoord, int primitiveId) const
    {
        return fragmentShader(fragment, fragPos, texCoord, primitiveId);
    }
};

KERNEL void renderKernel(const float4* __restrict__ depthBufferColor, float4* __restrict__ renderTarget, int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        const int index     = y * width + x;
        renderTarget[index] = depthBufferColor[index];
    }
}

KERNEL void compositeF2BKernel(float4* __restrict__ depthBufferColor, int width, int height, float4* __restrict__ clearColor)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        const int index = y * width + x;
        float4& f       = depthBufferColor[index];

        // Front-to-back alpha blending composite (blend the fragment over the clear color)
        float4 outColor;
        const float4& dstColor             = clearColor[0];
        const float4& srcColor             = f;
        const float3 premultipliedDstColor = make_float3(dstColor.x * dstColor.w, dstColor.y * dstColor.w, dstColor.z * dstColor.w);
        const float3 premultipliedSrcColor = make_float3(srcColor.x * srcColor.w, srcColor.y * srcColor.w, srcColor.z * srcColor.w);
        const float outAlpha               = srcColor.w + dstColor.w * (1.0f - srcColor.w);
        outColor.x                         = fma(premultipliedDstColor.x, (1.0f - srcColor.w), premultipliedSrcColor.x);
        outColor.y                         = fma(premultipliedDstColor.y, (1.0f - srcColor.w), premultipliedSrcColor.y);
        outColor.z                         = fma(premultipliedDstColor.z, (1.0f - srcColor.w), premultipliedSrcColor.z);
        outColor.w                         = outAlpha;

        if (outAlpha > 0.0f)
        {
            outColor.x /= outAlpha;
            outColor.y /= outAlpha;
            outColor.z /= outAlpha;
        }

        f = outColor;
    }
}

namespace roesti
{
    Rasterizer2D::Rasterizer2D()
        : mConfigs{}
        , mData{}
    {
    }

    Rasterizer2D::~Rasterizer2D()
    {
    }

    void Rasterizer2D::createPipeline(const PipelineConfig& config, int index)
    {
        CRITICAL_ASSERT(index >= 0 && index < MAX_PIPELINES, "Pipeline index %d is out of bounds [0, %d).", index, MAX_PIPELINES);

        mConfigs[index] = config;

        CHECK_TENSOR(mConfigs[index].outputRT, 3, 4, torch::kFloat32);
        // output rt has to have width amd height
        CRITICAL_ASSERT(mConfigs[index].outputRT.size(0) == mConfigs[index].height, "Output render target height %d does not match config height %d.", mConfigs[index].outputRT.size(0), mConfigs[index].height);
        CRITICAL_ASSERT(mConfigs[index].outputRT.size(1) == mConfigs[index].width, "Output render target width %d does not match config width %d.", mConfigs[index].outputRT.size(1), mConfigs[index].width);
        CRITICAL_ASSERT(mConfigs[index].width > 0 && mConfigs[index].height > 0, "Width and height must be greater than zero.");

        mData.depthBufferColor = torch::zeros({ mConfigs[index].height, mConfigs[index].width, 4 }, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        mData.depthBufferLayer = torch::zeros({ mConfigs[index].height, mConfigs[index].width }, torch::dtype(torch::kInt32).device(torch::kCUDA));

        int numTilesX      = (mConfigs[index].width + TILE_WIDTH - 1) / TILE_WIDTH;
        int numTilesY      = (mConfigs[index].height + TILE_HEIGHT - 1) / TILE_HEIGHT;
        int totalTileCount = numTilesX * numTilesY;

        mData.tileRanges = torch::zeros({ totalTileCount, 2 }, torch::dtype(torch::kInt32).device(torch::kCUDA));

        printf("Created 2D rasterizer pipeline with size %dx%d, primitive type %d, blend mode %d at index %d.\n",
               mConfigs[index].width,
               mConfigs[index].height,
               static_cast<int>(mConfigs[index].primitiveType),
               static_cast<int>(mConfigs[index].blendMode),
               index);
    }

    void Rasterizer2D::bindPipeline(int index)
    {
        CHECK_TENSOR(mData.depthBufferColor, 3, 4, torch::kFloat32);
        CHECK_TENSOR1(mData.depthBufferLayer, 2, torch::kInt32);
        CHECK_TENSOR(mData.tileRanges, 2, 2, torch::kInt32);
        CRITICAL_ASSERT(index >= 0 && index < MAX_PIPELINES, "Pipeline index %d is out of bounds [0, %d).", index, MAX_PIPELINES);
        CRITICAL_ASSERT(mConfigs[index].outputRT.defined(), "Pipeline at index %d is not created. Call createPipeline() first.", index);

        mEnabledPipeline = index;
    }

    void Rasterizer2D::addVertexBuffer(const torch::Tensor& vertexPositions, const torch::Tensor& vertexColors, const torch::Tensor& vertexTexcoords, const torch::Tensor& vertexLayers, int vertexCount, int index)
    {
        CHECK_TENSOR(vertexPositions, 2, 2, torch::kFloat32);
        CHECK_TENSOR(vertexColors, 2, 4, torch::kFloat32);
        CHECK_TENSOR(vertexTexcoords, 2, 2, torch::kFloat32);
        CHECK_TENSOR(vertexLayers, 1, vertexPositions.size(0), torch::kInt32);
        CRITICAL_ASSERT(vertexPositions.size(0) == vertexCount, "Vertex positions tensor size %d does not match vertex count %d.", vertexPositions.size(0), vertexCount);
        CRITICAL_ASSERT(vertexColors.size(0) == vertexCount, "Vertex colors tensor size %d does not match vertex count %d.", vertexColors.size(0), vertexCount);
        CRITICAL_ASSERT(vertexTexcoords.size(0) == vertexCount, "Vertex texcoords tensor size %d does not match vertex count %d.", vertexTexcoords.size(0), vertexCount);
        CRITICAL_ASSERT(vertexLayers.size(0) == vertexCount, "Vertex layers tensor size %d does not match vertex count %d.", vertexLayers.size(0), vertexCount);
        CRITICAL_ASSERT(index >= 0 && index < MAX_VBOS, "VBO index %d is out of bounds [0, %d).", index, MAX_VBOS);

        VertexBufferObject& vbo = mData.vertexBufferObjects[index];

        vbo.vertexInPositions = vertexPositions;
        vbo.vertexInColors    = vertexColors;
        vbo.vertexInTexcoords = vertexTexcoords;
        vbo.vertexInLayers    = vertexLayers;

        vbo.vertexOutPositions = torch::zeros({ vertexCount, 2 }, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        vbo.vertexOutColors    = torch::zeros({ vertexCount, 4 }, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        vbo.vertexOutTexcoords = torch::zeros({ vertexCount, 2 }, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        vbo.vertexOutLayers    = torch::zeros({ vertexCount }, torch::dtype(torch::kInt32).device(torch::kCUDA));

        vbo.vertexCount = vertexCount;
    }

    void Rasterizer2D::addIndexBuffer(const torch::Tensor& indexBuffer, int indexCount, int index)
    {
        CHECK_TENSOR1(indexBuffer, 1, torch::kInt32);
        CRITICAL_ASSERT(indexBuffer.size(0) == indexCount, "Index buffer tensor size %d does not match index count %d.", indexBuffer.size(0), indexCount);
        CRITICAL_ASSERT(index >= 0 && index < MAX_VBOS, "VBO index %d is out of bounds [0, %d).", index, MAX_VBOS);

        VertexBufferObject& vbo = mData.vertexBufferObjects[index];

        vbo.indexBuffer = indexBuffer;

        vbo.indexCount = indexCount;
    }

    const int Rasterizer2D::MAX_VBOS;
    const int Rasterizer2D::MAX_PIPELINES;

    void Rasterizer2D::bindVertexBufferObject(int index)
    {
        CRITICAL_ASSERT(mEnabledPipeline >= 0, "No pipeline is currently enabled. Call bindPipeline() first.");
        CRITICAL_ASSERT(index >= 0 && index < MAX_VBOS, "VBO index %d is out of bounds [0, %d).", index, MAX_VBOS);
        CRITICAL_ASSERT(mData.vertexBufferObjects[index].vertexCount > 0, "Vertex buffer object at index %d is not created. Call addVertexBuffer() first.", index);
        CRITICAL_ASSERT(mData.vertexBufferObjects[index].indexCount > 0, "Vertex buffer object at index %d is not created. Call addIndexBuffer() first.", index);

        mEnabledVBO = index;

        VertexBufferObject& vbo = mData.vertexBufferObjects[index];

        vbo.primitiveCount = 0;

        if (mConfigs[mEnabledPipeline].primitiveType == PrimitiveType::Points)
            vbo.primitiveCount = vbo.indexCount;
        else if (mConfigs[mEnabledPipeline].primitiveType == PrimitiveType::Lines)
            vbo.primitiveCount = vbo.indexCount / 2;
        else if (mConfigs[mEnabledPipeline].primitiveType == PrimitiveType::LineStrip)
            vbo.primitiveCount = vbo.indexCount > 1 ? vbo.indexCount - 1 : 0;
        else if (mConfigs[mEnabledPipeline].primitiveType == PrimitiveType::Triangles)
            vbo.primitiveCount = vbo.indexCount / 3;
        else if (mConfigs[mEnabledPipeline].primitiveType == PrimitiveType::TriangleStrip)
            vbo.primitiveCount = vbo.indexCount > 2 ? vbo.indexCount - 2 : 0;

        // Only reallocate if not allocated or size is incorrect and smaller - bigger is always safe (torch does this internally anyway)
        vbo.primitivePositions = torch::zeros({ vbo.primitiveCount, 3, 2 }, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        vbo.primitiveColors    = torch::zeros({ vbo.primitiveCount, 3, 4 }, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        vbo.primitiveTexcoords = torch::zeros({ vbo.primitiveCount, 3, 2 }, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        vbo.primitiveLayers    = torch::zeros({ vbo.primitiveCount, 3 }, torch::dtype(torch::kInt32).device(torch::kCUDA));
        vbo.primitiveAABBs     = torch::full({ vbo.primitiveCount, 4 }, -1.0f, torch::dtype(torch::kFloat32).device(torch::kCUDA)); // initialized to -1 to indicate invalid AABB

        vbo.tilesTouched = torch::zeros({ vbo.primitiveCount }, torch::dtype(torch::kInt32).device(torch::kCUDA));
        vbo.depths       = torch::zeros({ vbo.primitiveCount }, torch::dtype(torch::kInt32).device(torch::kCUDA));
    }

    void Rasterizer2D::setClear(const torch::Tensor& clearColor, int clearLayer)
    {
        CHECK_TENSOR(clearColor, 1, 4, torch::kFloat32);
        CRITICAL_ASSERT(clearLayer >= 0, "Clear layer must be non-negative.");

        mData.clearColor = clearColor;
        mData.clearLayer = clearLayer;
    }

    void Rasterizer2D::rasterizeTiled()
    {
        const PipelineConfig& config = mConfigs[mEnabledPipeline];
        VertexBufferObject& vbo      = mData.vertexBufferObjects[mEnabledVBO];

        int numTilesX      = (config.width + TILE_WIDTH - 1) / TILE_WIDTH;
        int numTilesY      = (config.height + TILE_HEIGHT - 1) / TILE_HEIGHT;
        int totalTileCount = numTilesX * numTilesY;

        // compute for each primitive the number of tiles it touches
        dim3 blockSizeCTT(256);
        dim3 gridSizeCTT((vbo.primitiveCount + blockSizeCTT.x - 1) / blockSizeCTT.x);
        computeTilesTouchedKernel<<<gridSizeCTT, blockSizeCTT>>>(
            reinterpret_cast<const float4*>(vbo.primitiveAABBs.contiguous().data_ptr<float>()),
            reinterpret_cast<const int3*>(vbo.primitiveLayers.contiguous().data_ptr<int>()),
            vbo.primitiveCount,
            config.primitiveType,
            vbo.tilesTouched.contiguous().data_ptr<int>(),
            vbo.depths.contiguous().data_ptr<int>(),
            config.width,
            config.height);
        CUDA_CHECK_THROW(cudaPeekAtLastError());

        torch::Tensor cumulativeTilesTouched = torch::cumsum(vbo.tilesTouched, 0, torch::kInt32);

        int totalTouchedTiles = cumulativeTilesTouched.size(0) > 0
                                    ? cumulativeTilesTouched[-1].item<int>()
                                    : 0;

        mData.tileLookupKeys   = torch::zeros({ totalTouchedTiles }, torch::dtype(torch::kInt64).device(torch::kCUDA));
        mData.tileLookupValues = torch::zeros({ totalTouchedTiles }, torch::dtype(torch::kInt32).device(torch::kCUDA));

        dim3 blockSizeCTL(256);
        dim3 gridSizeCTL((vbo.primitiveCount + blockSizeCTL.x - 1) / blockSizeCTL.x);
        createTileLookupKernel<<<gridSizeCTL, blockSizeCTL>>>(
            reinterpret_cast<const float4*>(vbo.primitiveAABBs.contiguous().data_ptr<float>()),
            vbo.primitiveCount,
            cumulativeTilesTouched.contiguous().data_ptr<int>(),
            vbo.depths.contiguous().data_ptr<int>(),
            make_int2(numTilesX, numTilesY),
            config.width,
            config.height,
            mData.tileLookupKeys.contiguous().data_ptr<int64_t>(),
            mData.tileLookupValues.contiguous().data_ptr<int>());
        CUDA_CHECK_THROW(cudaPeekAtLastError());

        // sort tileLookupKeys and tileLookupValues based on tileLookupKeys
        auto sorted                 = torch::sort(mData.tileLookupKeys, -1L, /*descending=*/true);
        mData.tileLookupKeys        = std::get<0>(sorted);
        torch::Tensor sortedIndices = std::get<1>(sorted);
        mData.tileLookupValues      = torch::gather(mData.tileLookupValues, 0, sortedIndices);

        // delete unused sortedIndices
        sortedIndices.reset();

        // get tile ranges
        dim3 blockSizeGTL(256);
        dim3 gridSizeGTL((totalTouchedTiles + blockSizeGTL.x - 1) / blockSizeGTL.x);
        getTileRangesKernel<<<gridSizeGTL, blockSizeGTL>>>(
            mData.tileLookupKeys.contiguous().data_ptr<int64_t>(),
            totalTouchedTiles,
            reinterpret_cast<int2*>(mData.tileRanges.contiguous().data_ptr<int>()));
        CUDA_CHECK_THROW(cudaPeekAtLastError());

        dim3 blockSizeR(TILE_WIDTH, TILE_HEIGHT);
        dim3 gridSizeR(numTilesX, numTilesY);
        if (config.primitiveType == PrimitiveType::Points)
        {
            rasterizePointsTiledKernel<FragmentShaderFunctor><<<gridSizeR, blockSizeR>>>(
                reinterpret_cast<float2*>(vbo.primitivePositions.contiguous().data_ptr<float>()),
                reinterpret_cast<float4*>(vbo.primitiveColors.contiguous().data_ptr<float>()),
                reinterpret_cast<float2*>(vbo.primitiveTexcoords.contiguous().data_ptr<float>()),
                vbo.primitiveLayers.contiguous().data_ptr<int>(),
                vbo.primitiveCount,
                reinterpret_cast<float4*>(mData.depthBufferColor.contiguous().data_ptr<float>()),
                mData.depthBufferLayer.contiguous().data_ptr<int>(),
                config.blendMode,
                FragmentShaderFunctor(),
                config.pointSize,
                config.width,
                config.height,
                make_int2(numTilesX, numTilesY),
                reinterpret_cast<int2*>(mData.tileRanges.contiguous().data_ptr<int>()),
                mData.tileLookupValues.contiguous().data_ptr<int>());
        }
        else if (config.primitiveType == PrimitiveType::Lines || config.primitiveType == PrimitiveType::LineStrip)
        {
            rasterizeLinesTiledKernel<FragmentShaderFunctor><<<gridSizeR, blockSizeR>>>(
                reinterpret_cast<float2*>(vbo.primitivePositions.contiguous().data_ptr<float>()),
                reinterpret_cast<float4*>(vbo.primitiveColors.contiguous().data_ptr<float>()),
                reinterpret_cast<float2*>(vbo.primitiveTexcoords.contiguous().data_ptr<float>()),
                vbo.primitiveLayers.contiguous().data_ptr<int>(),
                vbo.primitiveCount,
                reinterpret_cast<float4*>(mData.depthBufferColor.contiguous().data_ptr<float>()),
                mData.depthBufferLayer.contiguous().data_ptr<int>(),
                config.blendMode,
                FragmentShaderFunctor(),
                config.lineWidth,
                config.width,
                config.height,
                make_int2(numTilesX, numTilesY),
                reinterpret_cast<int2*>(mData.tileRanges.contiguous().data_ptr<int>()),
                mData.tileLookupValues.contiguous().data_ptr<int>());
        }
        else if (config.primitiveType == PrimitiveType::Triangles || config.primitiveType == PrimitiveType::TriangleStrip)
        {
            rasterizeTrianglesTiledKernel<FragmentShaderFunctor><<<gridSizeR, blockSizeR>>>(
                reinterpret_cast<float2*>(vbo.primitivePositions.contiguous().data_ptr<float>()),
                reinterpret_cast<float4*>(vbo.primitiveColors.contiguous().data_ptr<float>()),
                reinterpret_cast<float2*>(vbo.primitiveTexcoords.contiguous().data_ptr<float>()),
                vbo.primitiveLayers.contiguous().data_ptr<int>(),
                vbo.primitiveCount,
                reinterpret_cast<float4*>(mData.depthBufferColor.contiguous().data_ptr<float>()),
                mData.depthBufferLayer.contiguous().data_ptr<int>(),
                config.blendMode,
                FragmentShaderFunctor(),
                config.lineWidth,
                config.wireframe,
                config.width,
                config.height,
                make_int2(numTilesX, numTilesY),
                reinterpret_cast<int2*>(mData.tileRanges.contiguous().data_ptr<int>()),
                mData.tileLookupValues.contiguous().data_ptr<int>());
        }
        CUDA_CHECK_THROW(cudaPeekAtLastError());
    }

    void Rasterizer2D::assemblePrimitives()
    {
        const PipelineConfig& config = mConfigs[mEnabledPipeline];
        VertexBufferObject& vbo      = mData.vertexBufferObjects[mEnabledVBO];

        dim3 blockSizeAP(256);
        dim3 gridSizeAP((vbo.primitiveCount + blockSizeAP.x - 1) / blockSizeAP.x);
        if (config.primitiveType == PrimitiveType::Points)
        {
            assemblePointsKernel<<<gridSizeAP, blockSizeAP>>>(
                reinterpret_cast<const float2*>(vbo.vertexOutPositions.contiguous().data_ptr<float>()),
                reinterpret_cast<const float4*>(vbo.vertexOutColors.contiguous().data_ptr<float>()),
                reinterpret_cast<const float2*>(vbo.vertexOutTexcoords.contiguous().data_ptr<float>()),
                vbo.vertexOutLayers.contiguous().data_ptr<int>(),
                vbo.indexBuffer.contiguous().data_ptr<int>(),
                reinterpret_cast<float2*>(vbo.primitivePositions.contiguous().data_ptr<float>()),
                reinterpret_cast<float4*>(vbo.primitiveColors.contiguous().data_ptr<float>()),
                reinterpret_cast<float2*>(vbo.primitiveTexcoords.contiguous().data_ptr<float>()),
                vbo.primitiveLayers.contiguous().data_ptr<int>(),
                reinterpret_cast<float4*>(vbo.primitiveAABBs.contiguous().data_ptr<float>()),
                vbo.primitiveCount,
                config.primitiveType,
                config.width,
                config.height,
                config.pointSize);
        }
        else if (config.primitiveType == PrimitiveType::Lines || config.primitiveType == PrimitiveType::LineStrip)
        {
            assembleLinesKernel<<<gridSizeAP, blockSizeAP>>>(
                reinterpret_cast<const float2*>(vbo.vertexOutPositions.contiguous().data_ptr<float>()),
                reinterpret_cast<const float4*>(vbo.vertexOutColors.contiguous().data_ptr<float>()),
                reinterpret_cast<const float2*>(vbo.vertexOutTexcoords.contiguous().data_ptr<float>()),
                vbo.vertexOutLayers.contiguous().data_ptr<int>(),
                vbo.indexBuffer.contiguous().data_ptr<int>(),
                reinterpret_cast<float2*>(vbo.primitivePositions.contiguous().data_ptr<float>()),
                reinterpret_cast<float4*>(vbo.primitiveColors.contiguous().data_ptr<float>()),
                reinterpret_cast<float2*>(vbo.primitiveTexcoords.contiguous().data_ptr<float>()),
                vbo.primitiveLayers.contiguous().data_ptr<int>(),
                reinterpret_cast<float4*>(vbo.primitiveAABBs.contiguous().data_ptr<float>()),
                vbo.primitiveCount,
                config.primitiveType,
                config.width,
                config.height,
                config.lineWidth);
        }
        else if (config.primitiveType == PrimitiveType::Triangles || config.primitiveType == PrimitiveType::TriangleStrip)
        {
            assembleTrianglesKernel<<<gridSizeAP, blockSizeAP>>>(
                reinterpret_cast<const float2*>(vbo.vertexOutPositions.contiguous().data_ptr<float>()),
                reinterpret_cast<const float4*>(vbo.vertexOutColors.contiguous().data_ptr<float>()),
                reinterpret_cast<const float2*>(vbo.vertexOutTexcoords.contiguous().data_ptr<float>()),
                vbo.vertexOutLayers.contiguous().data_ptr<int>(),
                vbo.indexBuffer.contiguous().data_ptr<int>(),
                reinterpret_cast<float2*>(vbo.primitivePositions.contiguous().data_ptr<float>()),
                reinterpret_cast<float4*>(vbo.primitiveColors.contiguous().data_ptr<float>()),
                reinterpret_cast<float2*>(vbo.primitiveTexcoords.contiguous().data_ptr<float>()),
                vbo.primitiveLayers.contiguous().data_ptr<int>(),
                reinterpret_cast<float4*>(vbo.primitiveAABBs.contiguous().data_ptr<float>()),
                vbo.primitiveCount,
                config.primitiveType,
                config.width,
                config.height,
                config.lineWidth);
        }
        CUDA_CHECK_THROW(cudaPeekAtLastError());
    }

    void Rasterizer2D::drawIndexed(bool clear)
    {
        CRITICAL_ASSERT(mEnabledPipeline >= 0, "No pipeline is currently enabled. Call bindPipeline() first.");
        CRITICAL_ASSERT(mEnabledVBO >= 0 && mEnabledVBO < MAX_VBOS, "No vertex buffer object is currently enabled. Call bindVertexBufferObject() first.");

        const PipelineConfig& config  = mConfigs[mEnabledPipeline];
        const VertexBufferObject& vbo = mData.vertexBufferObjects[mEnabledVBO];

        if (clear)
        {
            torch::Tensor clearColorValue;
            if (config.blendMode == BlendMode::AlphaBlendF2B)
            {
                clearColorValue = torch::zeros({ 1, 4 }, torch::dtype(torch::kFloat32).device(torch::kCUDA));
            }
            else
            {
                clearColorValue = mData.clearColor;
            }

            dim3 blockSizeC(16, 16);
            dim3 gridSizeC(config.width / blockSizeC.x + 1, config.height / blockSizeC.y + 1);
            clearBuffers<<<gridSizeC, blockSizeC>>>(
                reinterpret_cast<float4*>(mData.depthBufferColor.contiguous().data_ptr<float>()),
                mData.depthBufferLayer.contiguous().data_ptr<int>(),
                config.width,
                config.height,
                reinterpret_cast<const float4*>(clearColorValue.data_ptr<float>()),
                mData.clearLayer);
            CUDA_CHECK_THROW(cudaPeekAtLastError());
        }

        // vertex shader
        dim3 blockSizeVS(256);
        dim3 gridSizeVS((vbo.vertexCount + blockSizeVS.x - 1) / blockSizeVS.x);
        vertexShaderKernel<<<gridSizeVS, blockSizeVS>>>(
            reinterpret_cast<const float2*>(vbo.vertexInPositions.contiguous().data_ptr<float>()),
            reinterpret_cast<const float4*>(vbo.vertexInColors.contiguous().data_ptr<float>()),
            reinterpret_cast<const float2*>(vbo.vertexInTexcoords.contiguous().data_ptr<float>()),
            vbo.vertexInLayers.contiguous().data_ptr<int>(),
            reinterpret_cast<float2*>(vbo.vertexOutPositions.contiguous().data_ptr<float>()),
            reinterpret_cast<float4*>(vbo.vertexOutColors.contiguous().data_ptr<float>()),
            reinterpret_cast<float2*>(vbo.vertexOutTexcoords.contiguous().data_ptr<float>()),
            vbo.vertexOutLayers.contiguous().data_ptr<int>(),
            vbo.vertexCount,
            config.width,
            config.height);
        CUDA_CHECK_THROW(cudaPeekAtLastError());

        int primitiveCount = vbo.primitiveCount;

        // assemble primitives
        assemblePrimitives();

        // rasterize (includes fragment shading)
        rasterizeTiled();

        if (clear && config.blendMode == BlendMode::AlphaBlendF2B)
        {
            // for alpha blending front to back, we need to composite the background afterwards
            dim3 blockSizeC(16, 16);
            dim3 gridSizeC(config.width / blockSizeC.x + 1, config.height / blockSizeC.y + 1);
            compositeF2BKernel<<<gridSizeC, blockSizeC>>>(
                reinterpret_cast<float4*>(mData.depthBufferColor.contiguous().data_ptr<float>()),
                config.width,
                config.height,
                reinterpret_cast<float4*>(mData.clearColor.data_ptr<float>()));
            CUDA_CHECK_THROW(cudaPeekAtLastError());
        }

        // render
        dim3 blockSize(16, 16);
        dim3 gridSize((config.width + blockSize.x - 1) / blockSize.x, (config.height + blockSize.y - 1) / blockSize.y);
        renderKernel<<<gridSize, blockSize>>>(
            reinterpret_cast<const float4*>(mData.depthBufferColor.contiguous().data_ptr<float>()),
            reinterpret_cast<float4*>(config.outputRT.contiguous().data_ptr<float>()),
            config.width,
            config.height);
        CUDA_CHECK_THROW(cudaPeekAtLastError());
    }
};
