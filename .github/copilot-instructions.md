# GitHub Copilot Instructions - FlashMaster

## Project Context

AI-powered flashcard learning platform built with Next.js 15, deployed to Azure Static Web Apps at https://flashmaster.shtrial.com.

## Architecture Constraints

**MANDATORY: Use shared Azure platform resources**

- Azure OpenAI: `shared-openai-eastus2` (gpt-5.1, text-embedding-3-large)
- Azure Storage: `stmahumsharedapps` (container: flashmaster)
- Azure Key Vault: `kv-mahum-shared-apps` (secrets)

**PROHIBITED: Never introduce**

- OpenAI public API (use Azure OpenAI)
- Supabase, Firebase (use Azure Storage)
- Google Gemini, Anthropic (use Azure OpenAI)
- AWS S3, GCS (use Azure Blob Storage)
- Hardcoded API keys (use environment variables)

## Code Generation Preferences

### Component Pattern

```typescript
// Prefer function components with hooks
export function MyComponent({ prop1, prop2 }: MyComponentProps) {
  const [state, setState] = useState<Type>(initialValue);
  
  return (
    <div className="flex flex-col gap-4">
      {/* Mobile-first responsive design */}
    </div>
  );
}
```

### API Route Pattern

```typescript
// app/api/my-route/route.ts
import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    // Validate input
    if (!body.field) {
      return NextResponse.json({ error: 'Missing field' }, { status: 400 });
    }
    // Process request
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('API error:', error);
    return NextResponse.json({ error: 'Internal error' }, { status: 500 });
  }
}
```

### Azure OpenAI Pattern

```typescript
import { azureOpenAI } from '@/lib/azureOpenAI';

const response = await azureOpenAI.chat.completions.create({
  model: process.env.AZURE_OPENAI_CHAT_DEPLOYMENT!,
  messages: [
    { role: 'system', content: 'System prompt' },
    { role: 'user', content: userInput },
  ],
});
```

### Azure Storage Pattern

```typescript
import { uploadFile } from '@/lib/azureStorage';

const url = await uploadFile(file, 'folder-name');
```

## Testing Requirements

- All new features MUST have Playwright E2E tests
- Test on desktop-chrome AND mobile-chrome projects
- Use role-based selectors: `page.getByRole('button', { name: 'Submit' })`
- Mock Azure APIs: `page.route('**/openai.azure.com/**', mockHandler)`
- Verify mobile keyboard doesn't obscure content

## Mobile-First Design

- All interactive elements: minimum 44x44px touch targets
- Test viewports: 375px (Pixel 5), 390px (iPhone 12), 1280px (Desktop)
- Use responsive Tailwind classes: `text-sm sm:text-base md:text-lg`
- Verify content visible with mobile keyboard open
- Test orientation changes (portrait ↔ landscape)

## Accessibility (WCAG 2.1 AA)

- Use semantic HTML: `<button>`, `<nav>`, `<main>`, `<article>`
- Add `aria-label` to icon-only buttons
- Ensure keyboard navigation: Tab, Enter, Escape
- Minimum 4.5:1 contrast ratio for text
- Test with screen readers

## TypeScript Style

- Strict mode enabled
- Prefer interfaces over types for props
- Export types alongside implementation
- Use `satisfies` for type narrowing
- No `any` types

## Styling Rules

- Tailwind CSS utility classes only (no inline styles)
- Use `cn()` helper for conditional classes
- Mobile-first responsive patterns
- Follow existing component structure in `src/components/`

## State Management

- Zustand for global state (see `src/store/srsStore.ts`)
- `useState` for local component state
- IndexedDB for offline persistence (via `lib/indexedDBStorageAdapter.ts`)

## Performance

- Use Next.js `<Image>` component for all images
- Lazy load below-the-fold components
- Implement loading states for async operations
- Keep bundle size minimal
- Cache Azure API responses when appropriate

## Common Patterns to Follow

- Error boundaries for graceful failures
- Loading spinners for async operations
- Toast notifications for user feedback
- Optimistic UI updates
- Proper form validation

---

**Last Updated**: November 18, 2025
