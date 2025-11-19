# FlashMaster - Agent Development Guide

## Architecture Overview

FlashMaster is a **Next.js 15 full-stack application** for AI-powered flashcard learning, deployed to Azure Static Web Apps.

### Tech Stack
- **Frontend**: Next.js 15.3.2, React 19, Radix UI, Tailwind CSS 4, Zustand
- **Backend**: Next.js API Routes (App Router)
- **Storage**: IndexedDB (local), Azure Blob Storage (future)
- **AI**: Azure OpenAI (gpt-5.1, text-embedding-3-large)
- **Deployment**: Azure Static Web App at flashmaster.shtrial.com

---

## Shared Azure Platform (MANDATORY)

**CRITICAL**: Always use shared resources from `rg-shared-ai`:

| Service | Resource | Endpoint |
|---------|----------|----------|
| **Azure OpenAI** | `shared-openai-eastus2` | https://shared-openai-eastus2.openai.azure.com/ |
| **Azure Storage** | `stmahumsharedapps` | https://stmahumsharedapps.blob.core.windows.net/ |
| **Key Vault** | `kv-mahum-shared-apps` | For secrets management |

### Environment Variables

```env
# Azure OpenAI (Primary AI Provider)
AZURE_OPENAI_ENDPOINT=https://shared-openai-eastus2.openai.azure.com/
AZURE_OPENAI_KEY=<from_key_vault>
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5.1
AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-large

# Azure Storage
AZURE_STORAGE_BLOB_ENDPOINT=https://stmahumsharedapps.blob.core.windows.net/
AZURE_STORAGE_CONNECTION_STRING=<from_key_vault>
AZURE_STORAGE_CONTAINER=flashmaster
```

---

## Code Patterns

### Azure OpenAI Integration

```typescript
// lib/azureOpenAI.ts
import { AzureOpenAI } from 'openai';

export const azureOpenAI = new AzureOpenAI({
  apiKey: process.env.AZURE_OPENAI_KEY!,
  endpoint: process.env.AZURE_OPENAI_ENDPOINT!,
  apiVersion: process.env.AZURE_OPENAI_CHAT_API_VERSION || '2025-01-01-preview',
});

export async function generateFlashcards(content: string) {
  const response = await azureOpenAI.chat.completions.create({
    model: process.env.AZURE_OPENAI_CHAT_DEPLOYMENT!,
    messages: [
      { role: 'system', content: 'Generate flashcards from the provided content. Return JSON array of {question, answer} pairs.' },
      { role: 'user', content },
    ],
    response_format: { type: 'json_object' },
  });
  
  return JSON.parse(response.choices[0].message.content!);
}
```

### Azure Storage Upload

```typescript
// lib/azureStorage.ts
import { BlobServiceClient } from '@azure/storage-blob';

const blobService = BlobServiceClient.fromConnectionString(
  process.env.AZURE_STORAGE_CONNECTION_STRING!
);

export async function uploadFile(file: File, folder: string = 'uploads') {
  const containerClient = blobService.getContainerClient(
    process.env.AZURE_STORAGE_CONTAINER!
  );
  
  const fileName = `${folder}/${Date.now()}-${file.name}`;
  const blockBlobClient = containerClient.getBlockBlobClient(fileName);
  
  await blockBlobClient.uploadData(await file.arrayBuffer(), {
    blobHTTPHeaders: { blobContentType: file.type },
  });
  
  return blockBlobClient.url;
}
```

### API Route Pattern

```typescript
// app/api/generate-flashcards/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { generateFlashcards } from '@/lib/azureOpenAI';

export async function POST(request: NextRequest) {
  try {
    const { content } = await request.json();
    
    if (!content || content.length < 10) {
      return NextResponse.json(
        { error: 'Content too short' },
        { status: 400 }
      );
    }
    
    const flashcards = await generateFlashcards(content);
    
    return NextResponse.json({ flashcards });
  } catch (error) {
    console.error('Flashcard generation failed:', error);
    return NextResponse.json(
      { error: 'Failed to generate flashcards' },
      { status: 500 }
    );
  }
}
```

---

## Prohibited Patterns

❌ **DO NOT USE:**
- OpenAI public API directly (use Azure OpenAI)
- Supabase, Firebase (use Azure Storage + future PostgreSQL)
- Google Gemini, Anthropic Claude (use Azure OpenAI)
- AWS S3, Google Cloud Storage (use Azure Blob Storage)
- Hardcoded API keys (use environment variables)

✅ **ALWAYS USE:**
- Shared Azure platform resources
- Environment variables from `.env.local` (dev) or Azure Portal (prod)
- Key Vault references for secrets in production

---

## Testing Guidelines

### E2E Tests (Playwright)

**Mobile-First Approach:**
- Test on Pixel 5 (375x667) and iPhone 12 (390x844)
- Minimum touch target size: 44x44px
- Verify keyboard doesn't obscure content
- Test orientation changes (portrait/landscape)

**Azure Mocking:**
```typescript
// tests/flashcard-generation.spec.ts
import { test, expect } from '@playwright/test';

test('generate flashcards with Azure OpenAI', async ({ page }) => {
  // Mock Azure OpenAI API
  await page.route('**/openai.azure.com/**', (route) => {
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        choices: [{
          message: {
            content: JSON.stringify({
              flashcards: [
                { question: 'What is AI?', answer: 'Artificial Intelligence' }
              ]
            })
          }
        }]
      }),
    });
  });
  
  await page.goto('/generate');
  await page.fill('textarea[name="content"]', 'AI is amazing');
  await page.click('button:has-text("Generate")');
  
  await expect(page.locator('text=What is AI?')).toBeVisible();
});
```

### Unit Tests

```typescript
// lib/__tests__/srsLogic.test.ts
import { calculateNextReview } from '../srsLogic';

describe('SRS Algorithm', () => {
  test('calculates next review date correctly', () => {
    const result = calculateNextReview(1, 2.5, new Date('2025-11-18'));
    expect(result.nextReviewDate).toBeAfter(new Date('2025-11-18'));
  });
});
```

---

## Development Workflow

### Local Development

```bash
# Install dependencies
pnpm install

# Copy environment template
cp .env.example .env.local

# Add your Azure keys to .env.local
# AZURE_OPENAI_KEY=...
# AZURE_STORAGE_CONNECTION_STRING=...

# Start dev server
pnpm dev

# Open http://localhost:3000
```

### Testing

```bash
# Run all E2E tests
pnpm test:e2e

# Run specific test
pnpm test:e2e tests/happy-path.spec.ts

# Run on specific browser
pnpm test:e2e --project=mobile-chrome

# Debug mode
pnpm test:e2e --debug
```

### Deployment

```bash
# Commit changes
git add .
git commit -m "feat: add flashcard generation"

# Push to main (triggers Azure Static Web Apps deployment)
git push origin main

# Monitor deployment at:
# https://github.com/shmindmaster/flashmaster/actions
```

---

## Code Style & Conventions

### Component Structure

```typescript
// components/FlashcardGenerator.tsx
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';

interface FlashcardGeneratorProps {
  onGenerated: (flashcards: Flashcard[]) => void;
}

export function FlashcardGenerator({ onGenerated }: FlashcardGeneratorProps) {
  const [content, setContent] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  const handleGenerate = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/generate-flashcards', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content }),
      });
      const { flashcards } = await response.json();
      onGenerated(flashcards);
    } catch (error) {
      console.error('Generation failed:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="flex flex-col gap-4">
      <Textarea
        value={content}
        onChange={(e) => setContent(e.target.value)}
        placeholder="Paste content to generate flashcards..."
        className="min-h-[200px]"
      />
      <Button onClick={handleGenerate} disabled={isLoading}>
        {isLoading ? 'Generating...' : 'Generate Flashcards'}
      </Button>
    </div>
  );
}
```

### TypeScript Conventions

- Use strict mode (enabled in `tsconfig.json`)
- Prefer interfaces over types for props
- Use `function` keyword for components (not arrow functions)
- Export types alongside implementation
- Use `satisfies` for type narrowing

### Styling

- Tailwind CSS utility classes only
- Use `cn()` helper for conditional classes
- No inline styles
- Follow mobile-first responsive patterns:
  ```tsx
  <div className="text-sm sm:text-base md:text-lg">
    Mobile: 14px, Tablet: 16px, Desktop: 18px
  </div>
  ```

---

## Accessibility (WCAG 2.1 AA)

- Use semantic HTML (`<button>`, `<nav>`, `<main>`, `<article>`)
- Add `aria-label` to icon-only buttons
- Ensure keyboard navigation works (Tab, Enter, Escape)
- Minimum 4.5:1 contrast ratio for text
- Touch targets minimum 44x44px on mobile
- Test with screen readers (VoiceOver, NVDA)

---

## Performance Optimization

- Use Next.js `<Image>` component for all images
- Lazy load below-the-fold components
- Use `loading="lazy"` for images
- Keep bundle size small (check with `pnpm build`)
- Use IndexedDB for offline data
- Implement service worker for offline support (future)

---

## Common Gotchas

1. **Azure OpenAI Rate Limits**: Implement retry logic with exponential backoff
2. **IndexedDB Quotas**: Monitor storage usage, implement cleanup
3. **Mobile Keyboard**: Use `window.visualViewport` to handle keyboard overlay
4. **SRS Algorithm**: Test edge cases (first review, long intervals)
5. **Offline Sync**: Handle conflicts when coming back online

---

## Resources

- [Next.js Docs](https://nextjs.org/docs)
- [Azure OpenAI Reference](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Playwright Docs](https://playwright.dev)
- [Radix UI Components](https://www.radix-ui.com)
- [Tailwind CSS](https://tailwindcss.com)

---

**Last Updated**: November 18, 2025
